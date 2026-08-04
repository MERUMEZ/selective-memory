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

СПИСОК ДВУЯЗЫЧНЫЙ. Английская половина была неполной: два десятка слов
без единого личного маркера — ни "i", ни "my", ни "me", — тогда как
русские "я", "мой", "меня" в списке стояли. То есть основной язык
библиотеки (модель по умолчанию английская) обслуживался хуже
второстепенного, и в каждом вопросе вроде "what is my daughter called"
половина ключевых слов была служебной.

Порог перекрытия всё равно задан ДОЛЕЙ (contradiction_min_overlap), а не
фактом совпадения: никакой список не бывает полным, и опираться на "есть
хоть одно общее слово" нельзя ни в одном языке.
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
