Edit a file using one or more search and replace blocks.

Each block should be formatted as follows:
```
<<<<<<< SEARCH
[original code to find]
=======
[new code to replace with]
>>>>>>> REPLACE
```

All blocks are resolved against the original file contents before any changes are written.
The tool fails without writing if any block is malformed, does not match, matches multiple locations,
or overlaps with another block. Resolved replacements are applied from the end of the file toward
the start to avoid offset shifts.

Matching is attempted in this order for each block:
1. Exact match.
2. Per-line stripped match for indentation and trailing whitespace differences.
3. Normalized line endings.
4. Normalized smart quotes.

Returns:
    A summary of the update made to the file
