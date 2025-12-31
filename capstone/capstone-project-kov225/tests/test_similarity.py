from similarity import calculate_similarity


def test_identical_names():
    gt = {"defendants": ["John"], "places": ["York"]}
    htr = {"defendants": ["John"], "places": ["York"], "plea_primary": None}
    s = calculate_similarity(gt, htr)
    # identical → should be very high
    assert s >= 90


def test_completely_different():
    gt = {"defendants": ["Alice"], "places": ["London"]}
    htr = {"defendants": ["Zzzxx"], "places": ["Nowhere"], "plea_primary": None}
    s = calculate_similarity(gt, htr)
    # unrelated names -->very low
    assert s <= 30


def test_partial_overlap():
    gt = {"defendants": ["John"], "places": []}
    htr = {"defendants": ["Jon"], "places": [], "plea_primary": None}
    s = calculate_similarity(gt, htr)
    # John vs Jon scores ~90 with RapidFuzz
    assert 70 <= s <= 95


def test_empty_htr():
    gt = {"defendants": ["John"], "places": ["York"]}
    htr = {"defendants": [], "places": [], "plea_primary": None}
    s = calculate_similarity(gt, htr)
    # empty HTR list should be very low
    assert s < 50


def test_size_penalty_on_large_htr_list():
    gt = {"defendants": ["John"], "places": []}

    htr_small = {
        "defendants": ["John"],
        "places": [],
        "plea_primary": None,
    }

    htr_large = {
        "defendants": ["John", "Alice", "Thomas", "Henry"],
        "places": [],
        "plea_primary": None,
    }

    s_small = calculate_similarity(gt, htr_small)
    s_large = calculate_similarity(gt, htr_large)

    # Larger list must score strictly lower due to size penalty
    assert s_large < s_small
