from img2text.formats.otsl import otsl_to_html


def test_simple_table():
    s = "<otsl><fcel>A<fcel>1<nl><fcel>B<fcel>2<nl></otsl>"
    assert otsl_to_html(s) == ("<table><tr><td>A</td><td>1</td></tr>"
                               "<tr><td>B</td><td>2</td></tr></table>")


def test_empty_cell_kept():
    s = "<otsl><fcel>A<ecel><nl><fcel>B<fcel>2<nl></otsl>"
    assert otsl_to_html(s).count("<td>") == 4


def test_lcel_merges_left():
    s = "<otsl><fcel>Header<lcel><nl><fcel>A<fcel>1<nl></otsl>"
    assert '<td colspan="2">Header</td>' in otsl_to_html(s)


def test_ucel_merges_up():
    s = "<otsl><fcel>Year<fcel>2015<nl><ucel><fcel>Oil<nl></otsl>"
    assert 'rowspan="2"' in otsl_to_html(s)


def test_loc_tags_dropped_and_text_escaped():
    s = "<otsl><loc_0><loc_0><fcel>R&amp;D<fcel>$1<nl></otsl>"
    out = otsl_to_html(s)
    assert "loc_" not in out and "R&amp;amp;D" in out
