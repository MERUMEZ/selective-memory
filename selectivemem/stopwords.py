# Copyright (C) 2026 MERUMEZ <selectivemem@gmail.com>
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License version 3 as
# published by the Free Software Foundation. See LICENSE for the full text.
#
# It is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE.
#
# A commercial licence is available for use in closed products — see
# COMMERCIAL.md.
"""
================================================================================
 STOPWORDS.PY — Слова, которые ничего не различают
================================================================================
Список служебных слов для выделения ключевых. Вынесен отдельно, потому что
им пользуются и словарь, и поиск, а держать его в одном из них значило бы
сделать второй зависимым от первого без всякой причины.

СПИСОК РУССКИЙ, и это ограничение, а не оплошность. На английском тексте
"the", "is", "my" остаются значимыми, пересечение ключевых слов всегда
больше нуля, и проверка "хотя бы одно общее слово" перестаёт защищать от
ложных вытеснений. Именно поэтому порог перекрытия в настройках задан
долей (contradiction_min_overlap), а не фактом совпадения.
================================================================================
"""

from typing import Set



STOP_WORDS: Set[str] = {
    "и", "в", "на", "с", "по", "к", "у", "из", "за", "от", "до", "для",
    "что", "как", "это", "то", "я", "ты", "он", "она", "мы", "вы", "они",
    "не", "но", "а", "же", "бы", "ли", "или", "тот", "его", "ее", "их",
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on",
    "and", "or", "but", "for", "with", "at", "by", "it", "this", "that",
}
