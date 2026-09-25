#!/usr/bin/env python3
"""Emit a print-exact checkerboard PDF for floor-geometry calibration.

The whole value of the board is that its squares are a known physical size, so
the one thing this must not do is let a printer rescale it. PDF geometry is in
points (1/72 inch) and is written here directly, with no library in between, so
what lands on the paper is what was asked for -- provided the print dialog is
set to 100% / "actual size" and not "fit to page". A ruler is printed beside
the board so that assumption can be checked with a tape before the board is
ever used.

Defaults are sized for this robot: at a 9.5 cm lens height and 30 degrees of
downward pitch, depth in the image collapses with range -- a 2.5 cm square is
31 px deep at 12 cm and 3 px deep at 45 cm. Depth is what constrains pitch, so
the board is meant to sit close, roughly 12 to 30 cm ahead, where every square
is still ten or more pixels deep.
"""
import argparse

MM = 72. / 25.4                      # points per millimetre
A4 = (210. * MM, 297. * MM)


def content(squares_x, squares_y, square_mm, page):
    """Page content stream: the board, a caption, and a printed ruler."""
    side = square_mm * MM
    width, height = squares_x * side, squares_y * side
    x0 = (page[0] - width) / 2.
    y0 = page[1] - height - 45. * MM          # leave a header band above

    out = []
    out.append('0 0 0 rg')
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2:
                continue                       # every other cell stays white
            x = x0 + col * side
            y = y0 + (squares_y - 1 - row) * side
            out.append('%.4f %.4f %.4f %.4f re f' % (x, y, side, side))

    # A thin frame, so a board cropped by the printer is obvious at a glance.
    out.append('0.5 w 0 0 0 RG')
    out.append('%.4f %.4f %.4f %.4f re S' % (x0, y0, width, height))

    def text(x, y, size, string):
        escaped = string.replace('\\', r'\\').replace('(', r'\(').replace(')', r'\)')
        out.append('BT /F1 %.1f Tf %.4f %.4f Td (%s) Tj ET' % (size, x, y, escaped))

    top = page[1] - 22. * MM
    text(x0, top, 13, 'Floor-geometry calibration board')
    text(x0, top - 6.5 * MM, 9.5,
         '%d x %d squares of %.1f mm  --  %d x %d interior corners for findChessboardCorners'
         % (squares_x, squares_y, square_mm, squares_x - 1, squares_y - 1))
    text(x0, top - 12. * MM, 9.5,
         'PRINT AT 100% / ACTUAL SIZE. Do not use "fit to page" or "shrink to fit".')

    # Ruler: check the print scale before trusting anything measured with this.
    ruler_y = y0 - 20. * MM
    ruler_len = 100. * MM
    out.append('0.8 w')
    out.append('%.4f %.4f m %.4f %.4f l S' % (x0, ruler_y, x0 + ruler_len, ruler_y))
    for i in range(11):
        x = x0 + i * 10. * MM
        tall = 4. * MM if i % 5 == 0 else 2.5 * MM
        out.append('%.4f %.4f m %.4f %.4f l S' % (x, ruler_y, x, ruler_y + tall))
    text(x0, ruler_y - 5.5 * MM, 9.,
         'This line is exactly 100 mm. Measure it. If it is not, the squares are not '
         '%.1f mm either.' % square_mm)

    text(x0, ruler_y - 13. * MM, 9.,
         'Lay flat on the floor, squares facing up, near edge about 12 cm ahead of the lens.')
    text(x0, ruler_y - 18. * MM, 9.,
         'Keep the whole board in frame and unshadowed; it does not need to be square to the robot.')
    return '\n'.join(out)


def pdf(stream, page):
    """Assemble a minimal one-page PDF with a correct cross-reference table."""
    objects = [
        '<< /Type /Catalog /Pages 2 0 R >>',
        '<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
        '<< /Type /Page /Parent 2 0 R /MediaBox [0 0 %.4f %.4f] '
        '/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>' % page,
        None,                                   # the content stream, below
        '<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
    ]
    body = stream.encode('ascii')
    objects[3] = '<< /Length %d >>\nstream\n%s\nendstream' % (len(body), stream)

    out = '%PDF-1.4\n'
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += '%d 0 obj\n%s\nendobj\n' % (number, obj)
    start = len(out)
    out += 'xref\n0 %d\n' % (len(objects) + 1)
    out += '0000000000 65535 f \n'
    for offset in offsets:
        out += '%010d 00000 n \n' % offset
    out += ('trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n'
            % (len(objects) + 1, start))
    return out.encode('latin-1')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--squares-x', type=int, default=7)
    parser.add_argument('--squares-y', type=int, default=5)
    parser.add_argument('--square-mm', type=float, default=25.)
    parser.add_argument('--out', default='calibration/checkerboard.pdf')
    args = parser.parse_args()

    side = args.square_mm * MM
    if args.squares_x * side > A4[0] - 20. * MM:
        raise SystemExit('board is wider than A4 at that square size')
    if args.squares_x == args.squares_y:
        raise SystemExit('use an asymmetric board so its orientation is unambiguous')

    data = pdf(content(args.squares_x, args.squares_y, args.square_mm, A4), A4)
    with open(args.out, 'wb') as handle:
        handle.write(data)
    print('%s  %d x %d squares of %.1f mm  (%.1f x %.1f cm board, %d x %d interior corners)'
          % (args.out, args.squares_x, args.squares_y, args.square_mm,
             args.squares_x * args.square_mm / 10., args.squares_y * args.square_mm / 10.,
             args.squares_x - 1, args.squares_y - 1))


if __name__ == '__main__':
    main()
