"""
BioSuite Ultra — Parallel Processing Utilities.

Provides thread pool and process pool executors for CPU-bound and I/O-bound
bioinformatics tasks. All implementations use Python stdlib (concurrent.futures)
— no external dependencies required.

Usage:
    from biosuite.core.parallel import parallel_map, parallel_apply

    # CPU-bound: uses ProcessPoolExecutor
    results = parallel_map(my_func, data_list, workers=4)

    # I/O-bound: uses ThreadPoolExecutor
    results = parallel_map(download_func, url_list, workers=8, io_bound=True)
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

from .log import get_logger

# get_logger() so the name follows the package (biosuite.core.parallel) and the
# shared handlers are installed on first use, like every other core module.
logger = get_logger(__name__)

# ── Auto-detect optimal worker count ──────────────────────────────────────

def get_optimal_workers(io_bound: bool = False) -> int:
    """Determine optimal number of workers based on system resources.
    
    Args:
        io_bound: If True, allow more workers (for I/O-bound tasks).
    
    Returns:
        Optimal number of worker threads/processes.
    """
    cpu_count = os.cpu_count() or 4
    if io_bound:
        return min(cpu_count * 2, 32)
    return min(cpu_count, 16)


# ── Parallel Map ──────────────────────────────────────────────────────────

def _is_picklable(func):
    """Check if a function can be pickled (needed for ProcessPoolExecutor on Windows)."""
    import pickle
    try:
        pickle.dumps(func)
        return True
    except (pickle.PicklingError, AttributeError, TypeError):
        return False


def _threads_preferred(funcs, io_bound: bool) -> bool:
    """Decide whether a thread pool is the right executor for *funcs*.

    A process pool pickles the callable into every worker, and on Windows (spawn)
    only an importable module-level function survives that (F2-5)."""
    if io_bound or sys.platform == 'win32':
        return True
    return not all(_is_picklable(func) for func in funcs)


def _run_sequential(func, items, kwargs) -> List[Any]:
    """Apply *func* to *items* in this process, mapping a failure to a logged None.

    parallel_map's pooled path already reports a failed item as None; the
    in-process paths have to agree, or the same call raises or returns None
    depending only on how many items it was given — which also makes
    ParallelBatchProcessor's 'failed' statistic unusable for short batches.
    """
    results = []
    for idx, item in enumerate(items):
        try:
            results.append(func(item, **kwargs))
        except Exception as e:
            logger.error("Parallel task failed at index %d: %s", idx, e)
            results.append(None)
    return results


def parallel_map(
    func: Callable,
    items: List[Any],
    workers: Optional[int] = None,
    io_bound: bool = False,
    chunk_size: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    **kwargs
) -> List[Any]:
    """Apply a function to all items in parallel.
    
    Uses ProcessPoolExecutor for CPU-bound tasks (default) and
    ThreadPoolExecutor for I/O-bound tasks.
    
    Args:
        func: Function to apply. Must be picklable for process pools.
        items: List of items to process.
        workers: Number of workers. None = auto-detect.
        io_bound: If True, use thread pool instead of process pool.
        chunk_size: Items per batch, submitted as one task; works for either pool.
            None (the default) submits one task per item — there is no auto value.
        progress_callback: Optional callback(completed, total) for progress.
        **kwargs: Extra keyword arguments passed to func.
    
    Returns:
        List of results in the same order as input items.
    
    Example:
        >>> from biosuite.core.parallel import parallel_map
        >>> results = parallel_map(gc_content, sequences, workers=4)
    """
    if not items:
        return []

    if chunk_size is not None and (not isinstance(chunk_size, int)
                                   or isinstance(chunk_size, bool) or chunk_size < 1):
        # range(0, len(items), -5) is empty, so no task was ever submitted and the
        # call answered [None] * len(items) — every item "failed" silently.
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size!r}.")
    
    if workers is None:
        workers = get_optimal_workers(io_bound)
    
    # For small inputs, don't bother with parallelism
    if len(items) <= 4 or workers <= 1:
        return _run_sequential(func, items, kwargs)
    
    results = [None] * len(items)
    completed = 0
    
    # Windows spawn needs picklable module-level functions.  On POSIX,
    # non-picklable callables (closures, lambdas, locals — e.g. the helper
    # inside parallel_align_pairs) previously blew up inside the pickling
    # machinery of every worker and silently fell back to SEQUENTIAL mode
    # after an exception storm; proactive picklability check picks threads
    # instead, which is cheap and correct for such callables.
    if _threads_preferred([func], io_bound):
        executor_class = ThreadPoolExecutor
    else:
        executor_class = ProcessPoolExecutor
    
    try:
        with executor_class(max_workers=workers) as executor:
            # Submit all tasks
            future_to_idx = {}
            if chunk_size:
                # Chunk items for better process pool performance
                for i in range(0, len(items), chunk_size):
                    chunk = items[i:i + chunk_size]
                    future = executor.submit(_apply_chunk, func, chunk, kwargs)
                    # (start, item count, submitted-as-a-chunk).  The count is the
                    # REAL chunk length, so a short final chunk cannot drive
                    # `completed` past len(items) in the progress callback.
                    future_to_idx[future] = (i, len(chunk), True)
            else:
                for idx, item in enumerate(items):
                    future = executor.submit(func, item, **kwargs)
                    future_to_idx[future] = (idx, 1, False)

            # Collect results
            for future in as_completed(future_to_idx):
                idx, size, chunked = future_to_idx[future]
                try:
                    # No timeout here: as_completed only yields futures that are
                    # already done, so a timeout on result() could never fire —
                    # the old timeout=300 was inert decoration.
                    result = future.result()
                    if chunked:
                        # _apply_chunk returns exactly one value per item of the
                        # chunk it was handed, so idx + j is always in range.
                        for j, r in enumerate(result):
                            results[idx + j] = r
                    else:
                        results[idx] = result
                    completed += size
                except Exception as e:
                    if chunked and size > 1:
                        # The whole chunk died on one item, so blaming a single
                        # index pointed the reader at an item that ran fine.
                        logger.error("Parallel task failed for items %d-%d: %s",
                                     idx, idx + size - 1, e)
                    else:
                        logger.error("Parallel task failed at index %d: %s", idx, e)
                    # Fill with None for failed items
                    for j in range(size):
                        results[idx + j] = None
                    completed += size
                # Outside the try, so a failed batch reports its progress too:
                # a callback wired to a GUI bar used to stall below 100% whenever
                # the last submitted unit of work raised.
                if progress_callback:
                    progress_callback(completed, len(items))
    except Exception as e:
        logger.error("Parallel executor failed: %s. Falling back to sequential.", e)
        return _run_sequential(func, items, kwargs)
    
    return results


def _apply_chunk(func, chunk, kwargs):
    """Apply function to a chunk of items."""
    return [func(item, **kwargs) for item in chunk]


# ── Parallel Submit (for heterogeneous tasks) ─────────────────────────────

def parallel_submit(
    tasks: List[tuple],
    workers: Optional[int] = None,
    io_bound: bool = False,
    timeout: float = 600.0
) -> List[Any]:
    """Submit heterogeneous tasks in parallel.
    
    Args:
        tasks: List of (func, args, kwargs) tuples.
        workers: Number of workers. None = auto-detect.
        io_bound: If True, use thread pool.
        timeout: Max seconds to wait for all tasks.
    
    Returns:
        List of results in the same order as input tasks.
    
    Example:
        >>> tasks = [
        ...     (gc_content, ("ATCG",), {}),
        ...     (reverse_complement, ("ATCG",), {}),
        ...     (translate, ("ATGAAATTT",), {"frame": 1}),
        ... ]
        >>> results = parallel_submit(tasks, workers=2)
    """
    if not tasks:
        return []
    
    if workers is None:
        workers = get_optimal_workers(io_bound)
    
    if len(tasks) <= 1 or workers <= 1:
        results = []
        for func, args, kwargs in tasks:
            try:
                results.append(func(*args, **kwargs))
            except Exception as e:
                logger.error("Task failed: %s", e)
                results.append(None)
        return results
    
    results = [None] * len(tasks)
    # The same rule parallel_map applies: a process pool cannot carry a
    # non-picklable callable, and on Windows (spawn) it cannot carry one that is
    # only reachable from the parent's __main__.  Sending such a task costs the
    # whole task, so heterogeneous batches fall back to threads instead.
    if _threads_preferred([func for func, *_rest in tasks], io_bound):
        executor_class = ThreadPoolExecutor
    else:
        executor_class = ProcessPoolExecutor
    
    try:
        with executor_class(max_workers=workers) as executor:
            future_to_idx = {}
            for idx, (func, args, kwargs) in enumerate(tasks):
                future = executor.submit(func, *args, **kwargs)
                future_to_idx[future] = idx
            
            for future in as_completed(future_to_idx, timeout=timeout):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error("Task %d failed: %s", idx, e)
                    results[idx] = None
    except Exception as e:
        logger.error("Parallel executor failed: %s", e)
        results = []
        for func, args, kwargs in tasks:
            try:
                results.append(func(*args, **kwargs))
            except Exception:
                results.append(None)
    
    return results


# ── Parallel Batch Processor ──────────────────────────────────────────────

class ParallelBatchProcessor:
    """Process items in parallel batches with progress tracking.
    
    Useful for large datasets that need to be processed in chunks.
    
    Example:
        >>> processor = ParallelBatchProcessor(workers=4)
        >>> results = processor.process(my_func, large_list, batch_size=100)
    """
    
    def __init__(self, workers: Optional[int] = None, io_bound: bool = False):
        self.workers = workers or get_optimal_workers(io_bound)
        self.io_bound = io_bound
        self.stats = {'total': 0, 'completed': 0, 'failed': 0, 'time': 0.0}
    
    def process(
        self,
        func: Callable,
        items: List[Any],
        batch_size: int = 100,
        progress_callback: Optional[Callable] = None,
        **kwargs
    ) -> List[Any]:
        """Process items in batches.
        
        Args:
            func: Function to apply to each item.
            items: List of items to process.
            batch_size: Number of items per batch.
            progress_callback: Optional progress callback.
            **kwargs: Extra keyword arguments for func.
        
        Returns:
            List of results.
        """
        self.stats['total'] = len(items)
        self.stats['completed'] = 0
        self.stats['failed'] = 0
        start_time = time.time()
        
        all_results = []
        
        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start:batch_start + batch_size]
            batch_results = parallel_map(
                func, batch,
                workers=self.workers,
                io_bound=self.io_bound,
                **kwargs
            )
            all_results.extend(batch_results)
            
            self.stats['completed'] += len(batch)
            self.stats['failed'] += sum(1 for r in batch_results if r is None)
            
            if progress_callback:
                progress_callback(self.stats['completed'], self.stats['total'])
        
        self.stats['time'] = time.time() - start_time
        return all_results
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return self.stats.copy()


# ── Convenience Functions ─────────────────────────────────────────────────

def parallel_gc_content(sequences: List[str], workers: Optional[int] = None) -> List[float]:
    """Calculate GC content for multiple sequences in parallel.
    
    Args:
        sequences: List of DNA sequences.
        workers: Number of workers (None = auto).
    
    Returns:
        List of GC content percentages.
    """
    from biosuite.core.sequence import gc_content
    return parallel_map(gc_content, sequences, workers=workers)


def parallel_reverse_complement(sequences: List[str], workers: Optional[int] = None) -> List[str]:
    """Compute reverse complement for multiple sequences in parallel.
    
    Args:
        sequences: List of DNA sequences.
        workers: Number of workers (None = auto).
    
    Returns:
        List of reverse complemented sequences.
    """
    from biosuite.core.sequence import reverse_complement
    return parallel_map(reverse_complement, sequences, workers=workers)


def parallel_translate(sequences: List[str], frame: int = 1, workers: Optional[int] = None) -> List[str]:
    """Translate multiple DNA sequences in parallel.
    
    Args:
        sequences: List of DNA sequences.
        frame: Reading frame (1-3, -1 to -3).
        workers: Number of workers (None = auto).
    
    Returns:
        List of protein sequences.
    """
    from biosuite.core.sequence import translate
    return parallel_map(translate, sequences, workers=workers, frame=frame)


def parallel_align_pairs(
    pairs: List[tuple],
    algorithm: str = 'needleman_wunsch',
    workers: Optional[int] = None
) -> List[tuple]:
    """Align multiple sequence pairs in parallel.
    
    Args:
        pairs: List of (seq1, seq2) tuples.
        algorithm: 'needleman_wunsch' or 'smith_waterman'.
        workers: Number of workers (None = auto).
    
    Returns:
        List of (aligned_seq1, aligned_seq2, score) tuples.
    """
    from biosuite.core.alignment import needleman_wunsch, smith_waterman
    
    func = needleman_wunsch if algorithm == 'needleman_wunsch' else smith_waterman
    
    def align_pair(pair):
        return func(pair[0], pair[1])
    
    return parallel_map(align_pair, pairs, workers=workers)


# ── Module-level lazy singleton for worker pool ───────────────────────────

_default_pool = None

def get_default_pool(workers: Optional[int] = None, io_bound: bool = False):
    """Get or create a default worker pool for reuse.
    
    Returns a ParallelBatchProcessor instance.
    """
    global _default_pool
    if _default_pool is None:
        _default_pool = ParallelBatchProcessor(workers=workers, io_bound=io_bound)
    return _default_pool


def shutdown_default_pool():
    """Shutdown the default worker pool."""
    global _default_pool
    _default_pool = None
