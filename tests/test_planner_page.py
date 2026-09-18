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
                # A compound selector names several classes at once
                # ('.stage.alt'); every part has to exist, and checking the
                # whole string as one name reports a fault that is only in
                # the check.
                for name in selector[1:].split('.'):
                    self.assertIn(name, classes, selector)
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
        # every click and every drawn marker in the wrong place. Both views are
        # on screen now, so both have to be checked: the second one carries the
        # same floor points and is just as wrong if it is sized to the other
        # picture.
        text = page()
        self.assertIn('pad.width = mode===\'camera\' ? 640 : 480', text)
        self.assertIn('apad.width = altIsFloor ? 480 : 640', text)
        css = styles().replace(' ', '')
        self.assertRegex(css, r'\.stage\.camera\{width:640px;height:480px\}')
        self.assertRegex(css, r'\.stage\.alt\{width:640px;height:480px\}')
        self.assertRegex(css, r'\.stage\.alt\.floorview\{width:480px;height:480px\}')

    def test_the_two_views_always_show_different_things(self):
        # The second view is defined as the one you are not drawing on. If both
        # ever resolved to the same mode the page would show the same picture
        # twice and the comparison the pair exists for would be gone.
        text = page()
        self.assertIn("const altIsFloor = mode==='camera';", text)
        self.assertIn("'/api/frame?view='+(altIsFloor?'floor':'camera')", text)

    def test_both_pictures_refresh_from_the_same_capture(self):
        # One picture lagging leaves markers drawn over floor the robot has
        # already left, which reads exactly like a reprojection fault.
        self.assertIn('function refreshShots()', page())
        self.assertNotRegex(
            script(), r"getElementById\('shot'\)\.src='/api/frame\?t='\+Date\.now\(\);"
                      r"(?!.*altshot)")

    def test_the_drawing_surface_is_not_stretched_by_its_own_controls(self):
        # The left column must be pinned to the picture, or a wide control row
        # expands it and squeezes the log panel off the screen.
        # 480 + 10 gap + 640 = 1130: the two pictures it holds. Pinned, so a
        # wide control row still wraps inside it instead of stretching it and
        # squeezing the log panel off the screen; max-width lets it shrink on
        # a narrow display rather than forcing the page to scroll sideways.
        self.assertRegex(styles().replace(' ', ''), r'\.pane\{flex:0 0 auto;?'
                         .replace(' ', '') + r'width:1130px;max-width:100%\}')

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


class MemorySiteTests(unittest.TestCase):
    """Every memory the loops build must carry the same things.

    This is a source check rather than a behaviour test because the bug it
    guards against is duplication, not logic: the loops build their memory in
    seven places, and twice now a field was added to some of them and missed on
    others. A turn between two steps of an instruction then erased the record of
    what had already been reached, and the robot went back to the thing it had
    just arrived at.
    """

    def source(self):
        with open(os.path.join(ROOT, 'local_nav/planner_demo.py')) as handle:
            return handle.read()

    def test_every_memory_carries_progress_and_the_goal(self):
        # Not a regex over the call: [^)]* stops at the bracket inside
        # list(obstacles) and silently checks a fragment. Take the text up to
        # the closing bracket that balances the opening one.
        text = self.source()
        found = 0
        for start in range(len(text)):
            if not text.startswith('memory = dict(', start):
                continue
            depth, index = 0, start + len('memory = dict')
            while index < len(text):
                if text[index] == '(':
                    depth += 1
                elif text[index] == ')':
                    depth -= 1
                    if depth == 0:
                        break
                index += 1
            call = ' '.join(text[start:index + 1].split())
            if 'dict( self.memory' in call or 'dict(self.memory' in call:
                # Derived from the memory it replaces, so everything not named
                # here is carried through by dict() itself.
                continue
            if 'route=' not in call or 'target=' in call:
                # ASK GPT keeps a memory too, but it plans one route on demand
                # rather than running a sequence, so it has no progress to carry
                # and no goal to chase. It is keyed by target=, which the loop
                # memories are not.
                continue
            found += 1
            self.assertIn('done=', call,
                          'a memory without the progress list: %s' % call)
            self.assertIn('goal=', call,
                          'a memory without the goal: %s' % call)
        self.assertGreaterEqual(found, 5, 'memory sites not found')


class BothViewsTests(unittest.TestCase):
    """Anything that sends one view's points must send the other's.

    The page draws two pictures from one answer. The reply to a command and the
    progress poll are separate code paths, and the poll is the one running
    while the robot drives -- so a field added to the reply and missed on the
    poll leaves the second view frozen exactly when it is being watched. That
    has already happened once in this file with the obstacle circles.
    """

    def source(self):
        with open(os.path.join(ROOT, 'local_nav/planner_demo.py')) as handle:
            return handle.read()

    def test_no_caller_computes_the_pair_for_itself(self):
        # Two callers, one definition. A third call to drawable() outside the
        # helper is how the paths drifted apart last time.
        text = self.source()
        self.assertIn('def both_views(self):', text)
        outside = text.count('self.drawable(') + text.count('bench.drawable(')
        self.assertEqual(outside, 2,
                         'drawable() is called outside both_views(); the two '
                         'response paths will drift')

    def test_every_response_carrying_points_also_carries_the_other_view(self):
        text = self.source()
        for kind, marker in (('reply', "out['points'] = points"),
                             ('poll', 'points=route, hazards=hazards')):
            self.assertIn(marker, text, kind)
        # The poll is the one that gets forgotten, so name it explicitly.
        poll = text[text.index("if path == '/api/progress'"):]
        poll = poll[:poll.index('return self._send') + 400]
        for field in ('alt_points', 'alt_hazards', 'alt_mode'):
            self.assertIn(field, poll,
                          'the progress poll omits %s, so the second view '
                          'freezes while the robot drives' % field)


class UnseenGoalTests(unittest.TestCase):
    """An unseen goal must never throw away the route the model just gave.

    The prompt is explicit that with a last_time the model must NOT return an
    empty route -- it carries on along the one it gave before, because
    following it is what brings the object back into view, and that matters
    most exactly when the object cannot be seen. ASK GPT used to discard the
    answer whenever visible was false, so the log said "continuing prior route
    toward the right" and the floor stayed empty. The two driving loops never
    had the bug: for them `visible` gates the goal, not the plan.
    """

    def source(self):
        with open(os.path.join(ROOT, 'local_nav/planner_demo.py')) as handle:
            return handle.read()

    def test_no_path_bails_out_on_an_unseen_goal(self):
        text = self.source()
        self.assertNotIn("return self.state(points=[], hazards=[], length=0.)",
                         text,
                         'a path still discards the answer when the goal is '
                         'not in view; the model was told to keep planning')

    def route_blocks(self):
        """The code in each path that turns route_pixels into floor points."""
        text = self.source()
        found = []
        at = 0
        while True:
            at = text.find("for point in answer.get('route_pixels') or []:", at)
            if at < 0:
                return found
            stop = text.index('project_obstacles', at)
            found.append(text[at:stop])
            at = stop

    def test_building_the_route_never_consults_visible(self):
        # A source check aimed at the block itself rather than at lines that
        # begin with the test: gating the loop -- `... or []) if visible else
        # []` -- is the same bug written sideways, and a check that only looks
        # at `if` lines walks straight past it.
        blocks = self.route_blocks()
        self.assertGreaterEqual(len(blocks), 2, 'route building not found')
        for block in blocks:
            self.assertNotIn('visible', block,
                             'the route is only built when the goal is in '
                             'view; the model was told to keep planning '
                             'without a sighting')

    def test_the_prompt_still_demands_a_route_without_a_sighting(self):
        # If this instruction ever goes, the executor change above becomes
        # pointless rather than wrong -- but silently.
        import fetch
        self.assertIn('DO NOT return an empty route', fetch.PROMPT)


class DrawnPriorTests(unittest.TestCase):
    """What is drawn on the canvas has to reach the model."""

    def source(self):
        with open(os.path.join(ROOT, 'local_nav/planner_demo.py')) as handle:
            return handle.read()

    def test_a_drawn_route_is_offered_when_there_is_no_remembered_one(self):
        text = self.source()
        self.assertIn('if prior is None and self.live_route:', text,
                      'a hand-drawn route is never sent to the model')

    def test_the_remembered_plan_still_wins(self):
        # The model's own leftover route carries target_was, which a drawn one
        # cannot. Preferring the drawn route would lose the bearing to a goal
        # that has gone out of frame -- the thing that makes an unseen target
        # recoverable at all.
        text = self.source()
        remembered = text.index("prior = fetch.recall(self.robot, self.memory")
        drawn = text.index('if prior is None and self.live_route:')
        self.assertLess(remembered, drawn,
                        'the drawn route is consulted before the remembered '
                        'plan; it would mask target_was')


class PriorKeyTests(unittest.TestCase):
    """Nothing may read a key out of a prior that recall() does not produce.

    The prior's shape has changed three times -- obstacles added, then the
    target, then obstacles removed because the model can see them for itself --
    and each time a reader was left behind. The last one raised KeyError inside
    ASK GPT, which is a crash in the operator's hands rather than a test
    failure. This compares what recall() returns against what the loops read.
    """

    def keys_produced(self):
        sys.path.insert(0, os.path.join(ROOT, 'local_nav'))
        import fetch
        import evaluate_gpt_routes as helper
        lens = helper.Lens()
        lens.pixel = fetch.Robot.pixel.__get__(lens)
        prior = fetch.recall(lens, dict(route=[(0., 40.)],
                                        obstacles=[('x', (5., 20.))],
                                        goal=(0., 60.), done=['a']),
                             [0., 10., 0.])
        self.assertIsNotNone(prior, 'recall produced nothing to check against')
        return set(prior)

    def test_every_key_read_from_a_prior_is_one_recall_returns(self):
        with open(os.path.join(ROOT, 'local_nav/planner_demo.py')) as handle:
            source = handle.read()
        read = set(re.findall(r"prior\['([a-z_]+)'\]", source))
        read |= set(re.findall(r"prior\.get\('([a-z_]+)'", source))
        self.assertTrue(read, 'no prior accesses found; the check is broken')
        self.assertEqual(read - self.keys_produced(), set(),
                         'planner_demo reads prior keys recall does not return')


if __name__ == '__main__':
    unittest.main()
