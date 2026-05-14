"""Utilities for parsing partial JSON during streaming."""

import json
import re
from typing import Optional, Tuple


class PartialJSONParser:
    """Parser for extracting values from partial JSON strings during streaming."""

    @staticmethod
    def extract_string_field(partial_json: str, field_name: str) -> Tuple[Optional[str], bool]:
        """
        Extract a string field value from partial JSON.

        Args:
            partial_json: Partial JSON string that may be incomplete
            field_name: Name of the field to extract

        Returns:
            Tuple of (field_value, is_complete)
            - field_value: The extracted string value or None if not found
            - is_complete: Whether the field value is complete (closing quote found)
        """
        # Find the field name first
        field_pattern = rf'"{field_name}"\s*:\s*"'
        field_match = re.search(field_pattern, partial_json)

        if not field_match:
            return None, False

        # Start extracting after the opening quote
        start_pos = field_match.end()
        value = ""
        pos = start_pos
        escaped = False

        # Parse character by character to handle escapes properly
        while pos < len(partial_json):
            char = partial_json[pos]

            if escaped:
                # This character is escaped, add it as-is
                value += char
                escaped = False
            elif char == "\\":
                # This is an escape character
                escaped = True
                value += char
            elif char == '"':
                # Found the closing quote (unescaped)
                return value, True
            else:
                # Regular character
                value += char

            pos += 1

        # Reached end of string without finding closing quote
        return value, False

    @staticmethod
    def extract_create_file_content(partial_json: str) -> Tuple[Optional[str], Optional[str], bool]:
        """
        Extract content and relative_path fields from partial create_file tool JSON.

        Args:
            partial_json: Partial JSON string for create_file tool input

        Returns:
            Tuple of (relative_path, content, is_content_complete)
        """
        # First try to get the relative_path (usually comes first)
        relative_path, _ = PartialJSONParser.extract_string_field(partial_json, "relative_path")

        # Then try to get the content field
        content, is_complete = PartialJSONParser.extract_string_field(partial_json, "content")

        # Handle escaped characters in content if we have any
        # The content already contains the raw escape sequences from JSON
        if content:
            # Process escape sequences in the correct order
            # Use a simple state machine to handle escapes properly
            result = ""
            i = 0
            while i < len(content):
                if content[i] == "\\" and i + 1 < len(content):
                    next_char = content[i + 1]
                    if next_char == "n":
                        result += "\n"
                        i += 2
                    elif next_char == "t":
                        result += "\t"
                        i += 2
                    elif next_char == "r":
                        result += "\r"
                        i += 2
                    elif next_char == '"':
                        result += '"'
                        i += 2
                    elif next_char == "\\":
                        result += "\\"
                        i += 2
                    elif next_char == "/":
                        result += "/"
                        i += 2
                    else:
                        # Unknown escape, keep as-is
                        result += content[i]
                        i += 1
                else:
                    result += content[i]
                    i += 1
            content = result

        return relative_path, content, is_complete

    @staticmethod
    def is_valid_partial_json(partial_json: str) -> bool:
        """
        Check if the partial JSON string is structurally valid so far.

        Args:
            partial_json: Partial JSON string

        Returns:
            True if the partial JSON is structurally valid, False otherwise
        """
        # Count braces and brackets
        open_braces = partial_json.count("{")
        close_braces = partial_json.count("}")
        open_brackets = partial_json.count("[")
        close_brackets = partial_json.count("]")

        # Check that we don't have more closing than opening
        if close_braces > open_braces or close_brackets > open_brackets:
            return False

        # Check for unterminated strings (odd number of unescaped quotes)
        # This is a simplified check
        quotes = 0
        escaped = False
        for char in partial_json:
            if char == "\\" and not escaped:
                escaped = True
                continue
            if char == '"' and not escaped:
                quotes += 1
            escaped = False

        # If we have an odd number of quotes, we have an unterminated string
        # which is okay for partial JSON
        return True

    @staticmethod
    def try_complete_json(partial_json: str) -> Optional[dict]:
        """
        Try to parse partial JSON by completing it with closing brackets/braces.

        Args:
            partial_json: Partial JSON string

        Returns:
            Parsed JSON dict if successful, None otherwise
        """
        if not partial_json.strip():
            return None

        # Try to parse as-is first
        try:
            return json.loads(partial_json)
        except json.JSONDecodeError:
            pass

        # Count unclosed braces and brackets
        open_braces = partial_json.count("{")
        close_braces = partial_json.count("}")
        open_brackets = partial_json.count("[")
        close_brackets = partial_json.count("]")

        # Add missing closing characters
        completed = partial_json

        # Close any unterminated strings first
        # Check if the last non-whitespace character is inside a string
        stripped = partial_json.rstrip()
        if stripped:
            # Simple check: if we have an odd number of quotes, add a closing quote
            quotes = 0
            escaped = False
            for char in stripped:
                if char == "\\" and not escaped:
                    escaped = True
                    continue
                if char == '"' and not escaped:
                    quotes += 1
                escaped = False

            if quotes % 2 == 1:
                completed += '"'

        # Add missing brackets and braces
        completed += "]" * (open_brackets - close_brackets)
        completed += "}" * (open_braces - close_braces)

        # Try to parse the completed JSON
        try:
            return json.loads(completed)
        except json.JSONDecodeError:
            return None
