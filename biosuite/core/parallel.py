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
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Callable, List, Any, Optional, Dict
from functools import partial

logger = logging.getLogger('biosuite.parallel')

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
        chunk_size: Items per chunk for process pool (None = auto).
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
    
    if workers is None:
        workers = get_optimal_workers(io_bound)
    
    # For small inputs, don't bother with parallelism
    if len(items) <= 4 or workers <= 1:
        return [func(item, **kwargs) for item in items]
    
    results = [None] * len(items)
    completed = 0
    
    # On Windows, ProcessPoolExecutor uses 'spawn' which requires picklable functions.
    # If io_bound or on Windows with non-module functions, use ThreadPoolExecutor.
    # On Windows, ProcessPoolExecutor uses 'spawn' which requires picklable
    # functions defined in importable modules. For safety, always use ThreadPoolExecutor
    # on Windows unless the user explicitly needs process-level parallelism.
    if io_bound or sys.platform == 'win32':
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
                    future_to_idx[future] = (i, chunk_size)
            else:
                for idx, item in enumerate(items):
                    future = executor.submit(func, item, **kwargs)
                    future_to_idx[future] = (idx, 1)
            
            # Collect results
            for future in as_completed(future_to_idx):
                idx, size = future_to_idx[future]
                try:
                    result = future.result(timeout=300)
                    if size == 1:
                        results[idx] = result
                    else:
                        # Chunk result
                        for j, r in enumerate(result):
                            if idx + j < len(items):
                                results[idx + j] = r
                    completed += size
                    if progress_callback:
                        progress_callback(completed, len(items))
                except Exception as e:
                    logger.error("Parallel task failed at index %d: %s", idx, e)
                    # Fill with None for failed items
                    for j in range(size):
                        if idx + j < len(items):
                            results[idx + j] = None
                    completed += size
    except Exception as e:
        logger.error("Parallel executor failed: %s. Falling back to sequential.", e)
        return [func(item, **kwargs) for item in items]
    
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
    executor_class = ThreadPoolExecutor if io_bound else ProcessPoolExecutor
    
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
