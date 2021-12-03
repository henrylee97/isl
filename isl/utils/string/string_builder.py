from __future__ import annotations


class Tab:
    def string(self, tab_size: int = 4) -> str:
        return ' ' * tab_size


class StringBuilder:

    def __init__(self, tab_size: int = 4) -> None:
        self._buffer = []
        self._indentation = 0
        self.tab_size = tab_size

    def indent(self) -> StringBuilder:
        self._indentation += 1
        return self

    def dedent(self) -> StringBuilder:
        self._indentation -= 1
        if self._indentation < 0:
            self._indentation = 0
        return self

    def _prepend_indentation(self) -> None:
        if len(self._buffer) == 0 or self._buffer[-1][-1] != '\n':
            return
        tabs = ' ' * self.tab_size
        for _ in range(self._indentation):
            self._buffer.append(tabs)

    def append_line(self, line: str) -> StringBuilder:
        self._prepend_indentation()
        self._buffer.append(line)
        self._buffer.append('\n')
        return self

    def append_word(self, word: str) -> StringBuilder:
        self._prepend_indentation()
        self._buffer.append(word)
        self._buffer.append(' ')
        return self

    def newline(self) -> StringBuilder:
        self._buffer.append('\n')
        return self

    def __str__(self) -> str:
        return ''.join(map(lambda x: x if not isinstance(x, Tab) else x.string(self.tab_size), self._buffer))
