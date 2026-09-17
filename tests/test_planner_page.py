"""Static checks on the bench page.

Both faults that reached the user in this page were visible without a browser
and neither was caught, because nothing looked at the markup at all:

  * a bare `img,canvas` positioning rule, written when the page had exactly one
    of each, later also caught the filmstrip in the side panel and stacked it on
    top of the live view;
  * `#strip{display:block}`, where an id selector outranks `[hidden]`, so
    setting `.hidden` on the element silently stopped hiding it.

Rendering the page needs a browser, and the one on this machine is unreliable.
These checks need nothing: they read the markup the server actually serves and
assert the handful of properties that class of bug violates.
"""

import os
import re
import sys
import unittest
from html.parser import HTMLParser

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'local_nav'))


def page():
    import planner_demo
    return planner_demo.PAGE


class Markup(HTMLParser):
    """Ids, classes and tags, and which element carries which."""

    def __init__(self):
        HTMLParser.__init__(self)
        self.ids = []
        self.elements = []          # (tag, id, [classes])
        self.open_tags = []
        self.unclosed = []

    VOID = {'img', 'input', 'br', 'hr', 'meta', 'link', 'source'}

    def handle_starttag(self, tag, attrs):
        got = dict(attrs)
        name = got.get('id')
        if name:
            self.ids.append(name)
        self.elements.append((tag, name, (got.get('class') or '').split()))
        if tag not in self.VOID:
            self.open_tags.append(tag)

    def handle_endtag(self, tag):
        if self.open_tags and self.open_tags[-1] == tag:
            self.open_tags.pop()
        elif tag in self.open_tags:
            self.unclosed.append(tag)
            while self.open_tags and self.open_tags.pop() != tag:
                pass


def parsed():
    m = Markup()
    m.feed(page())
    return m


def styles():
    """The text inside <style>, with comments stripped."""
    block = re.search(r'<style>(.*?)</style>', page(), re.S)
    return re.sub(r'/\*.*?\*/', '', block.group(1), flags=re.S) if block else ''


def script():
    return '\n'.join(re.findall(r'<script>(.*?)</script>', page(), re.S))


def rules():
    """(selector, body) for each CSS rule."""
    return [(sel.strip(), body)
            for sel, body in re.findall(r'([^{}]+)\{([^{}]*)\}', styles())]


class PageTests(unittest.TestCase):

    def test_every_id_the_script_reaches_for_exists(self):
        have = set(parsed().ids)
        wanted = set(re.findall(r"getElementById\(['\"]([^'\"]+)['\"]\)", script()))
        self.assertTrue(wanted, 'no element lookups found; the check is broken')
        self.assertEqual(wanted - have, set(),
                         'script reaches for ids the markup does not define')

    def test_every_selector_the_script_reaches_for_exists(self):
        classes = set()
        for _, _, names in parsed().elements:
            classes.update(names)
        for selector in re.findall(r"querySelector(?:All)?\(['\"]([^'\"]+)['\"]\)",
                                   script()):
            if selector.startswith('.'):
                self.assertIn(selector[1:], classes, selector)
            elif selector.startswith('#'):
                self.assertIn(selector[1:], parsed().ids, selector)

    def test_ids_are_unique(self):
        found = parsed().ids
        self.assertEqual(sorted(found), sorted(set(found)),
                         'duplicate id: getElementById would return one of them')

    def test_tags_are_balanced(self):
        m = parsed()
        self.assertEqual(m.unclosed, [])
        self.assertEqual(m.open_tags, [])

    def test_positioning_rules_are_scoped_to_a_container(self):
        # The original bug exactly: a bare tag selector that positions elements
        # absolutely will also catch any later element of that tag.
        for selector, body in rules():
            if 'position:absolute' not in body.replace(' ', ''):
                continue
            for part in selector.split(','):
                part = part.strip()
                self.assertTrue(
                    part.startswith('.') or part.startswith('#') or ' ' in part,
                    'bare tag selector %r positions absolutely; scope it to a '
                    'container or it will catch elements added later' % part)

    def test_no_id_rule_sets_display_and_defeats_hidden(self):
        # [hidden] has specificity (0,1,0); an id selector has (1,0,0) and wins,
        # so el.hidden = true stops working with no error anywhere.
        for selector, body in rules():
            for part in selector.split(','):
                part = part.strip()
                if part.startswith('#') and 'display:' in body.replace(' ', ''):
                    self.assertIn('[hidden]', styles(),
                                  '%s sets display; nothing restores [hidden]' % part)

    def test_hidden_is_forced_since_the_page_toggles_it(self):
        if '.hidden' in script() or 'hidden]' in script() or ' hidden>' in page():
            self.assertRegex(styles().replace(' ', ''),
                             r'\[hidden\]\{display:none!important\}')

    def test_the_canvas_matches_the_view_it_is_drawn_over(self):
        # A canvas whose backing store differs from the picture under it puts
        # every click and every drawn marker in the wrong place.
        text = page()
        self.assertIn('pad.width = mode===\'camera\' ? 640 : 480', text)
        self.assertRegex(styles().replace(' ', ''),
                         r'\.stage\.camera\{width:640px;height:480px\}')
        self.assertRegex(styles().replace(' ', ''), r'\.pane\.camera\{width:640px\}')

    def test_the_drawing_surface_is_not_stretched_by_its_own_controls(self):
        # The left column must be pinned to the picture, or a wide control row
        # expands it and squeezes the log panel off the screen.
        self.assertRegex(styles().replace(' ', ''), r'\.pane\{flex:0 0 auto;?'
                         .replace(' ', '') + r'width:480px\}')

    def test_every_endpoint_the_script_calls_is_served(self):
        with open(os.path.join(ROOT, 'local_nav/planner_demo.py')) as handle:
            source = handle.read()
        # The dot matters: /api/stream.mjpg is a real route, and a pattern that
        # stops at the dot reports a mismatch that is only in the pattern.
        served = set(re.findall(r"path == '(/api/[a-z_.]+)'", source))
        served |= set(re.findall(r"path\.startswith\('(/api/[a-z_.]+)'\)", source))
        called = set(re.findall(r"['\"](/api/[a-z_.]+)", script()))
        self.assertTrue(called)
        self.assertEqual(called - served, set(),
                         'the page calls endpoints the server does not route')


if __name__ == '__main__':
    unittest.main()
