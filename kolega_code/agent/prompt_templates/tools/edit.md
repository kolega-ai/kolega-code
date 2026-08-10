Edit a file using one search and replace block.

The block should be formatted as follows:
```
<<<<<<< SEARCH
[original code to find]
=======
[new code to replace with]
>>>>>>> REPLACE
```

Before using this tool:

1. Use the read tool to understand the file's contents and context.

To make a file edit, provide the following:
1. The path to the file to modify.
2. block: A single search and replace block.

The tool replaces one uniquely matched occurrence. Matching is attempted in this order:
1. Exact match.
2. Per-line stripped match for indentation and trailing whitespace differences.
3. Normalized line endings.
4. Normalized smart quotes.

CRITICAL REQUIREMENTS FOR USING THIS TOOL:

1. UNIQUENESS: The old_string MUST uniquely identify the specific instance you want to change. This means:
- Include AT LEAST 3-5 lines of context BEFORE the change point
- Include AT LEAST 3-5 lines of context AFTER the change point
- Include all whitespace, indentation, and surrounding code exactly as it appears in the file

2. SINGLE INSTANCE: This tool can only change ONE instance at a time. If you need to change multiple instances:
- Use multi_edit when all replacements are in the same file.
- Each block must uniquely identify its specific instance using extensive context.

3. VERIFICATION: Before using this tool:
- Check how many instances of the target text exist in the file
- If multiple instances exist, gather enough context to uniquely identify each one

WARNING: If you do not follow these requirements:
- The tool will fail if block matches multiple locations
- The tool will fail if block doesn't match after all fallback passes
- You may change the wrong instance if you don't include enough context

When making edits:
- Ensure the edit results in idiomatic, correct code
- Do not leave the code in a broken state

If you want to create or overwrite a file, use the write tool.

Returns:
    A summary of the update made to the file
