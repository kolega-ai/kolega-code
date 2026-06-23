import pytest
from bs4 import BeautifulSoup

from kolega_code.services.html import build_css_selector, extract_interactive_elements_from_html, get_associated_text


@pytest.fixture
def simple_html():
    return """
    <html>
    <body>
        <div id="content">
            <button id="submit-btn" class="primary" onclick="submitForm()">Submit</button>
            <a href="/about" class="nav-link">About Us</a>
            <input id="search" type="text" placeholder="Search..." name="q">
            <input id="password" type="password" value="secret">
            <label for="email">Email:</label>
            <input id="email" type="email" name="email">
            <select id="country" name="country">
                <option value="us">United States</option>
                <option value="ca">Canada</option>
                <option value="mx">Mexico</option>
            </select>
            <div role="button" data-target="modal">Open Modal</div>
            <textarea id="message" placeholder="Enter your message"></textarea>
            <div contenteditable="true">Editable content</div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def complex_html():
    return """
    <html>
    <body>
        <nav>
            <a href="/" class="home-link">Home</a>
            <a href="/products" class="nav-link">Products</a>
            <a href="/contact" class="nav-link">Contact</a>
        </nav>
        <div id="main">
            <form id="contact-form">
                <div class="form-group">
                    <label for="name">Name:</label>
                    <input id="name" type="text" name="name" placeholder="Your name">
                </div>
                <div class="form-group">
                    <label for="message">Message:</label>
                    <textarea id="message" name="message" placeholder="Your message"></textarea>
                </div>
                <div class="form-group">
                    <label for="country">Country:</label>
                    <select id="country" name="country">
                        <option value="">Select country</option>
                        <option value="us">United States</option>
                        <option value="uk">United Kingdom</option>
                    </select>
                </div>
                <div class="actions">
                    <button type="submit" class="btn-primary">Send</button>
                    <button type="reset" class="btn-secondary">Reset</button>
                </div>
            </form>
            <div class="modal" aria-hidden="true">
                <div class="modal-content">
                    <button class="close" aria-label="Close" data-toggle="modal">×</button>
                    <h2>Subscribe to Newsletter</h2>
                    <p>Get weekly updates about our products!</p>
                    <div role="button" class="subscribe-btn" onclick="subscribe()">Subscribe Now</div>
                </div>
            </div>
            <details>
                <summary>Read more</summary>
                <p>Additional information about our company.</p>
            </details>
        </div>
    </body>
    </html>
    """


class TestExtractInteractiveElements:
    def test_extract_all_element_types(self, simple_html):
        results = extract_interactive_elements_from_html(simple_html)

        # Check if we have the expected number of elements
        assert len(results) >= 9  # At least all the interactive elements we added

        # Check if we have all expected element types
        element_types = [result["element_type"] for result in results]
        assert "button" in element_types
        assert "a" in element_types
        assert "input" in element_types
        assert "select" in element_types
        assert "div" in element_types
        assert "textarea" in element_types

        # Check if specific elements are found
        submit_button = next((el for el in results if el["attributes"].get("id") == "submit-btn"), None)
        assert submit_button is not None
        assert submit_button["text"] == "Submit"
        assert "primary" in submit_button["attributes"].get("class", [])

        # Check link is found
        about_link = next((el for el in results if el["text"] == "About Us"), None)
        assert about_link is not None
        assert about_link["attributes"].get("href") == "/about"

        # Check select element has its options text
        country_select = next((el for el in results if el["attributes"].get("id") == "country"), None)
        assert country_select is not None
        assert "United States" in country_select["text"]
        assert "Canada" in country_select["text"]
        assert "Mexico" in country_select["text"]

        # Check div with role=button is found
        modal_button = next((el for el in results if el["attributes"].get("data-target") == "modal"), None)
        assert modal_button is not None
        assert modal_button["text"] == "Open Modal"

    def test_extract_from_complex_html(self, complex_html):
        results = extract_interactive_elements_from_html(complex_html)

        # Check if nav links are found
        nav_links = [el for el in results if el["element_type"] == "a"]
        assert len(nav_links) >= 3

        # Check if form elements are found
        form_inputs = [el for el in results if el["element_type"] == "input"]
        assert len(form_inputs) >= 1

        # Check if buttons are found
        buttons = [el for el in results if el["element_type"] == "button"]
        assert len(buttons) >= 3

        # Check if role=button is found
        role_buttons = [el for el in results if el["attributes"].get("role") == "button"]
        assert len(role_buttons) >= 1

        # Check if details/summary elements are found
        details_elements = [el for el in results if el["element_type"] in ("details", "summary")]
        assert len(details_elements) >= 1

    def test_handles_empty_html(self):
        results = extract_interactive_elements_from_html("")
        assert len(results) == 0

        results = extract_interactive_elements_from_html("<html><body></body></html>")
        assert len(results) == 0

    def test_handles_malformed_html(self):
        malformed_html = "<button>Incomplete Button<div>Not closed properly"
        results = extract_interactive_elements_from_html(malformed_html)

        # Even with malformed HTML, BeautifulSoup should find the button
        assert len(results) >= 1
        assert any(result["element_type"] == "button" for result in results)


class TestGetAssociatedText:
    def test_get_text_from_button(self):
        soup = BeautifulSoup("<button>Click Me</button>", "html.parser")
        element = soup.find("button")
        text = get_associated_text(element)
        assert text == "Click Me"

        # Test with value attribute
        soup = BeautifulSoup('<button value="Submit">Click Me</button>', "html.parser")
        element = soup.find("button")
        text = get_associated_text(element)
        assert text == "Click Me"  # Text content takes precedence

        # Test with empty button
        soup = BeautifulSoup("<button></button>", "html.parser")
        element = soup.find("button")
        text = get_associated_text(element)
        assert text == ""

    def test_get_text_from_input(self):
        # Test with placeholder
        soup = BeautifulSoup('<input placeholder="Enter name">', "html.parser")
        element = soup.find("input")
        text = get_associated_text(element)
        assert text == "Enter name"

        # Test with value (non-password)
        soup = BeautifulSoup('<input type="text" value="Default text">', "html.parser")
        element = soup.find("input")
        text = get_associated_text(element)
        assert text == "Default text"

        # Test with password type (should not return value)
        soup = BeautifulSoup('<input type="password" value="secret">', "html.parser")
        element = soup.find("input")
        text = get_associated_text(element)
        assert text == ""

        # Test with associated label
        soup = BeautifulSoup('<label for="username">Username:</label><input id="username">', "html.parser")
        element = soup.find("input")
        text = get_associated_text(element)
        assert text == "Username:"

    def test_get_text_from_link(self):
        soup = BeautifulSoup('<a href="/page">Visit Page</a>', "html.parser")
        element = soup.find("a")
        text = get_associated_text(element)
        assert text == "Visit Page"

        # Test with title attribute
        soup = BeautifulSoup('<a href="/page" title="Visit our page"></a>', "html.parser")
        element = soup.find("a")
        text = get_associated_text(element)
        assert text == "Visit our page"

    def test_get_text_from_select(self):
        soup = BeautifulSoup(
            """
            <select>
                <option>Option 1</option>
                <option>Option 2</option>
            </select>
        """,
            "html.parser",
        )
        element = soup.find("select")
        text = get_associated_text(element)
        assert "Option 1" in text
        assert "Option 2" in text

    def test_get_text_from_generic_element(self):
        soup = BeautifulSoup("<div>Some text</div>", "html.parser")
        element = soup.find("div")
        text = get_associated_text(element)
        assert text == "Some text"


class TestBuildCssSelector:
    def test_selector_with_id(self):
        soup = BeautifulSoup('<div id="unique-id">Content</div>', "html.parser")
        element = soup.find("div")
        selector = build_css_selector(element)
        assert selector == "#unique-id"

    def test_selector_with_class(self):
        # Simple class
        soup = BeautifulSoup('<div class="unique-class">Content</div>', "html.parser")
        element = soup.find("div")
        selector = build_css_selector(element)
        assert "div:nth-child" in selector

        # Multiple classes
        soup = BeautifulSoup('<div class="class1 class2">Content</div>', "html.parser")
        element = soup.find("div")
        selector = build_css_selector(element)
        assert "div:nth-child" in selector

    def test_selector_with_attributes(self):
        soup = BeautifulSoup('<input name="username" type="text">', "html.parser")
        element = soup.find("input")
        selector = build_css_selector(element)
        assert 'input[name="username"]' == selector

        # Test with data attribute
        soup = BeautifulSoup('<div data-testid="test-div">Content</div>', "html.parser")
        element = soup.find("div")
        selector = build_css_selector(element)
        assert selector == 'div[data-testid="test-div"]'

    def test_selector_with_nth_child(self):
        soup = BeautifulSoup(
            """
            <div>
                <p>First paragraph</p>
                <p>Second paragraph</p>
                <p>Third paragraph</p>
            </div>
        """,
            "html.parser",
        )
        elements = soup.find_all("p")

        # Second paragraph should use nth-child
        selector = build_css_selector(elements[1])
        assert "p:nth-child(2)" in selector

    def test_handles_special_characters(self):
        # Test with single quotes in attribute
        soup = BeautifulSoup('<div data-value="It\'s a test">Content</div>', "html.parser")
        element = soup.find("div")
        selector = build_css_selector(element)
        assert ":nth-child" in selector

        # Test with double quotes in attribute
        soup = BeautifulSoup("<div data-value='Say \"Hello\"'>Content</div>", "html.parser")
        element = soup.find("div")
        selector = build_css_selector(element)
        assert ":nth-child" in selector

    def test_handles_tailwind_classes(self):
        # Test with Tailwind-style class names containing colons
        soup = BeautifulSoup(
            '<div class="text-gray-200 hover:text-cyan-400 font-medium px-2">Tailwind styled div</div>', "html.parser"
        )
        element = soup.find("div")
        selector = build_css_selector(element)

        # The selector should either be the escaped class selector or fallback to a valid selector
        try:
            # Try to use the selector to verify it's valid
            soup.select(selector)

            # Check if it's using the class-based selector with escaped colons
            if selector.startswith("."):
                assert "\\:" in selector
        except Exception:
            # If it fell back to nth-child, that's acceptable too
            assert ":nth-child" in selector
