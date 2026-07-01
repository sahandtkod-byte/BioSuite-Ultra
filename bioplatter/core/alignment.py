"""
Pairwise and multiple alignment (Needleman-Wunsch, Smith-Waterman, MSA).
"""
import numpy as np

def _match_array(s1, s2, match, mismatch):
    """Precompute match/mismatch scores for all pairs."""
    n, m = len(s1), len(s2)
    eq = np.array([s1[i] == s2[j] for i in range(n) for j in range(m)]).reshape(n, m)
    return np.where(eq, match, mismatch)

def needleman_wunsch(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """Global alignment (Needleman-Wunsch). Returns (aligned_seq1, aligned_seq2, score)."""
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n+1, m+1), dtype=np.int32)
    dp[:, 0] = np.arange(n+1) * gap
    dp[0, :] = np.arange(m+1) * gap
    match_scores = _match_array(seq1, seq2, match, mismatch)
    for i in range(1, n+1):
        diag = dp[i-1, :m] + match_scores[i-1]
        up = dp[i-1, 1:] + gap
        left = dp[i, :m] + gap
        dp[i, 1:] = np.maximum(diag, np.maximum(up, left))
    # Traceback
    align1, align2 = [], []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (match if seq1[i-1]==seq2[j-1] else mismatch):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + gap:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1
    return ''.join(reversed(align1)), ''.join(reversed(align2)), dp[n][m]

def smith_waterman(seq1, seq2, match=1, mismatch=-1, gap=-2):
    """Local alignment (Smith-Waterman). Returns (aligned_seq1, aligned_seq2, score)."""
    n, m = len(seq1), len(seq2)
    dp = np.zeros((n+1, m+1), dtype=np.int32)
    max_score = 0
    max_pos = (0, 0)
    match_scores = _match_array(seq1, seq2, match, mismatch)
    for i in range(1, n+1):
        diag = dp[i-1, :m] + match_scores[i-1]
        up = dp[i-1, 1:] + gap
        left = dp[i, :m] + gap
        row = np.maximum(0, np.maximum(diag, np.maximum(up, left)))
        dp[i, 1:] = row
        row_max = np.max(row)
        if row_max > max_score:
            max_score = int(row_max)
            max_pos = (i, int(np.argmax(row)) + 1)
    # Traceback
    align1, align2 = [], []
    i, j = max_pos
    while i > 0 and j > 0 and dp[i][j] > 0:
        if dp[i][j] == dp[i-1][j-1] + (match if seq1[i-1]==seq2[j-1] else mismatch):
            align1.append(seq1[i-1])
            align2.append(seq2[j-1])
            i -= 1; j -= 1
        elif dp[i][j] == dp[i-1][j] + gap:
            align1.append(seq1[i-1])
            align2.append('-')
            i -= 1
        else:
            align1.append('-')
            align2.append(seq2[j-1])
            j -= 1
    return ''.join(reversed(align1)), ''.join(reversed(align2)), max_score

def multiple_alignment(sequences, method='clustal'):
    """
    Wrapper for external MSA (Clustal Omega or MUSCLE) if installed.
    Returns aligned sequences as list of strings.
    """
    raise NotImplementedError("Multiple alignment requires external tools (Clustal Omega). Install and add path.")