"""
================================================================================
 SETTINGS.PY — Настройки ядра памяти
================================================================================
Третий шаг выделения пакета: ядро больше НЕ ЧИТАЕТ глобальный config.

Раньше memory/* импортировал config.py — модуль на тысячу строк с
телеграм-токеном, промптами и настройками персонажа. Для библиотеки,
которую ставят через pip, это неприемлемо: у неё должна быть своя
конфигурация, а не зависимость от файла конкретного приложения.

Значения по умолчанию — те же, что были откалиброваны замерами (см.
AUDIT.md). Приложение может передать свои:

    graph = MemoryGraph(db=db, settings=MemorySettings(decay_rate=0.02))

или собрать из своего конфига целиком:

    graph = MemoryGraph(db=db, settings=MemorySettings.from_module(config))

Файл СГЕНЕРИРОВАН из config.py, чтобы значения гарантированно совпали:
переписывание семидесяти констант руками — верный способ незаметно
что-нибудь поменять. Тест test_settings.py сверяет их при каждом прогоне.
================================================================================
"""

from dataclasses import dataclass, fields
from typing import Any


@dataclass
class MemorySettings:
    """Параметры ядра памяти. Имена — те же, что в config.py, но в нижнем регистре."""

    age_t0: float = 25200.0
    babbling_syllable_pool_size: int = 30
    base_plasticity_threshold: float = 0.35
    concept_max_similar_links: int = 2
    concept_node_weight: float = 0.7
    concept_similarity_edge_weight: float = 0.25
    concept_similarity_link_threshold: float = 0.35
    concept_user_edge_weight: float = 0.4
    contradiction_correction_relief: float = 0.15
    contradiction_repeat_threshold: float = 0.85
    contradiction_stability_factor: float = 0.25
    contradiction_topic_threshold: float = 0.8
    contradiction_weight_penalty: float = 0.25
    decay_rate: float = 0.05
    default_user_model: str = 'Мой наставник и учитель (Юзер). Он занимается со мной, учит меня программированию, созданию игр и правильному общению.'
    edge_activation_decay: float = 0.6
    edge_activation_threshold: float = 0.3
    edge_boost_step: float = 0.15
    edge_decay_rate: float = 0.08
    edge_forget_threshold: float = 0.03
    edge_initial_weight: float = 0.2
    edge_max_hop_nodes: int = 3
    embeddings_enabled: bool = True
    embedding_model_path: str = '/var/www/mindnumbness/storage/models/navec_hudlit_v1_12B_500K_300d_100q.tar'
    forget_threshold: float = 0.05
    lexical_acquisition_enabled: bool = True
    lexical_age_t0: float = 2592000.0
    lexical_max_tokens_per_input: int = 20
    lexical_min_token_length: int = 2
    memory_fuzzy_weight: float = 0.1
    memory_keyword_weight: float = 0.3
    memory_min_keyword_length: int = 3
    memory_search_threshold: float = 0.3
    memory_semantic_weight: float = 0.5
    memory_weight_influence: float = 0.15
    retrospective_correction_enabled: bool = True
    retrospective_reversal_strength: float = 1.6
    retrospective_time_window_seconds: float = 180.0
    retrospective_window_size: int = 4
    reward_expectation_learning_rate: float = 0.3
    reward_min_learning_scale: float = 0.15
    reward_negative_penalty: float = 0.3
    reward_positive_boost: float = 0.2
    reward_positive_freshness_bonus: float = 600.0
    reward_preference_weight: float = 0.35
    sleep_abstract_node_weight: float = 0.75
    sleep_archive_weight_multiplier: float = 0.3
    sleep_hub_min_edge_weight: float = 0.5
    sleep_max_cluster_spokes: int = 2
    sleep_min_cluster_spokes: int = 2
    sleep_orphan_weight_threshold: float = 0.3
    stability_growth_factor: float = 1.5
    stability_initial: float = 1.0
    stability_max: float = 40.0
    stm_capacity: int = 16
    stm_emotional_threshold: float = 0.6
    stm_structural_threshold: float = 0.55
    stm_structural_weight: float = 0.5
    surprise_lexical_weight: float = 0.6
    surprise_structural_weight: float = 0.4
    syllable_node_initial_weight: float = 0.1
    syllable_node_reinforce_step: float = 0.03
    syllable_word_edge_weight: float = 0.45
    vocabulary_mastery_min_weight: float = 0.18
    word_cooccurrence_edge_weight: float = 0.12
    word_node_initial_weight: float = 0.15
    word_node_reinforce_step: float = 0.04

    # Путь к файлу базы. Это единственное поле, которое не является
    # параметром поведения памяти, — но конструктору Database он нужен,
    # а тащить ради него глобальный config незачем.
    db_path: str = "memory.db"

    # Имена, которые в config.py названы иначе. Список держится коротким
    # намеренно: расхождение имён — источник тихих рассогласований.
    _ALIASES = {"db_path": "BRAIN_DB_PATH"}

    @classmethod
    def from_module(cls, module: Any) -> "MemorySettings":
        """
        Собирает настройки из модуля с константами В ВЕРХНЕМ РЕГИСТРЕ.
        Отсутствующие поля берут значение по умолчанию — приложение вправе
        переопределить только часть.
        """
        values = {}
        for f in fields(cls):
            name = cls._ALIASES.get(f.name, f.name.upper())
            if hasattr(module, name):
                values[f.name] = getattr(module, name)
        return cls(**values)
