from typing import Dict, List

from bs4 import BeautifulSoup


def extract_interactive_elements_from_html(html_content: str) -> List[Dict]:
    """
    Extract interactive elements from HTML content with their associated text and selectors.

    Args:
        html_content (str): HTML content as a string

    Returns:
        List[Dict]: List of dictionaries containing element info with keys:
            - 'element_type': Type of the element (button, link, input, etc.)
            - 'selector': CSS selector to uniquely identify the element
            - 'text': Associated text with the element
            - 'attributes': Dictionary of element attributes
    """
    soup = BeautifulSoup(html_content, "html.parser")

    # Define interactive elements to look for
    interactive_elements = [
        # Links
        "a",
        # Form elements
        "button",
        "input",
        "select",
        "textarea",
        "option",
        # Interactive HTML5 elements
        "details",
        "summary",
        "dialog",
        # Elements that are often made interactive with JavaScript
        'div[role="button"]',
        'span[role="button"]',
        "[aria-controls]",
        "[onclick]",
        "[onmouseover]",
        "[onchange]",
        '[contenteditable="true"]',
        # Custom elements often used in frameworks
        "[data-toggle]",
        "[data-target]",
    ]

    results = []

    # Process each type of interactive element
    for selector in interactive_elements:
        try:
            if "[" in selector:
                # Handle attribute-based selectors
                tag_name, attr_part = selector.split("[", 1)
                attr_part = attr_part.rstrip("]")

                if "=" in attr_part:
                    # Attribute with specific value
                    attr_name, attr_value = attr_part.split("=", 1)
                    attr_value = attr_value.strip("\"'")
                    # Use find_all with attrs dict instead of select to avoid selector syntax issues
                    elements = soup.find_all(tag_name or True, attrs={attr_name: attr_value}) if tag_name else []
                    # Only try soup.select if we can't use the more reliable find_all method
                    if not elements and tag_name:
                        try:
                            elements = soup.select(selector)
                        except Exception:
                            # If select fails, continue with empty list
                            elements = []
                else:
                    # Just the presence of an attribute
                    elements = soup.find_all(lambda tag: tag.has_attr(attr_part))
            else:
                # Simple tag selector
                elements = soup.find_all(selector)
        except Exception as e:
            print(f"Warning: Error processing selector '{selector}': {e}")
            elements = []

        for element in elements:
            # Build a unique CSS selector for this element
            element_selector = build_css_selector(element)

            # Get the text associated with this element
            element_text = get_associated_text(element)

            # Get element type and attributes
            element_type = element.name
            attributes = {k: v for k, v in element.attrs.items()}

            # Add to results
            results.append(
                {
                    "element_type": element_type,
                    "selector": element_selector,
                    "text": element_text,
                    "attributes": attributes,
                }
            )

    return results


def get_associated_text(element) -> str:
    """
    Get text associated with an element. Handles different element types appropriately.
    """
    if element.name == "input":
        # For input elements, use placeholder, value or label text
        text = element.get("placeholder", "")
        if not text and element.get("value") and element["type"] not in ["password", "hidden"]:
            text = element["value"]

        # Try to find associated label
        element_id = element.get("id")
        if element_id:
            label = element.find_parent().find("label", attrs={"for": element_id})
            if label:
                return label.get_text(strip=True)

        return text

    elif element.name == "button":
        # Get button text or value
        return element.get_text(strip=True) or element.get("value", "")

    elif element.name == "a":
        # Get link text or title
        return element.get_text(strip=True) or element.get("title", "")

    elif element.name == "select":
        # For select elements, include the text of the options
        option_texts = [opt.get_text(strip=True) for opt in element.find_all("option")]
        return ", ".join(filter(None, option_texts))

    # Default: just return the text content
    return element.get_text(strip=True)


def build_css_selector(element) -> str:
    """
    Build a unique CSS selector for the given element.
    Tries to create the most specific but concise selector.
    Handles special characters and quotes in attribute values.
    """
    # If element has ID, that's the simplest and most reliable selector
    if element.get("id"):
        return f"#{element['id']}"

    # If element has classes, try to use them
    if element.get("class"):
        # Create individual class selectors and escape colons with backslashes
        escaped_classes = []
        for class_name in element["class"]:
            # Escape colons in class names (for utility frameworks like Tailwind)
            escaped_class = class_name.replace(":", "\\:")
            escaped_classes.append(escaped_class)

        class_selector = ".".join(escaped_classes)

        try:
            if len(element.select(f".{class_selector}")) == 1:
                return f".{class_selector}"
        except Exception:
            # If selector is invalid, fall back to other methods
            pass

    # Try with tag name and attribute combinations
    tag_name = element.name
    attributes = element.attrs

    for attr in ["name", "data-testid", "aria-label"]:
        if attr in attributes:
            attr_value = attributes[attr]
            # Escape single quotes and handle special characters in attribute values
            escaped_value = attr_value.replace("'", "\\'").replace('"', '\\"')
            # Use double quotes to avoid issues with apostrophes in the content
            selector = f'{tag_name}[{attr}="{escaped_value}"]'
            try:
                if len(element.find_parent().select(selector)) == 1:
                    return selector
            except Exception:
                # If the selector is invalid, try a simpler approach
                continue

    # Use nth-child as a last resort
    parents = []
    current = element

    # Build the path up to 3 levels of parents
    for _ in range(3):
        siblings = current.find_parent().find_all(current.name, recursive=False) if current.parent else []

        # Use 1-based index for nth-child
        if siblings:
            index = list(siblings).index(current) + 1
            selector_part = f"{current.name}:nth-child({index})"
        else:
            selector_part = current.name

        parents.append(selector_part)
        current = current.parent

        if current is None or current.name == "html":
            break

    # Construct the selector path
    selector = " > ".join(reversed(parents))

    # Validate the selector to make sure it's valid CSS
    try:
        # Test if the selector is valid by attempting to use it
        soup = BeautifulSoup("<html><body></body></html>", "html.parser")
        soup.select(selector)
        return selector
    except Exception:
        # If the selector is invalid, return a very simple one
        # This is a fallback that may not be unique but will be valid
        return f"{element.name}"
