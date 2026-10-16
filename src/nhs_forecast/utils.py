def calculate_growth(previous, current):
    """
    Calculate percentage change between two prescription volumes.
    Args:
        previous: Prescription volume in the previous month.
        current: Prescription volume in the current month.
    Returns:
        Percentage change as a float.
    Raises:
        ValueError: If previous is zero.
    """
    if previous == 0:
        raise ValueError("Previous prescription volume cannot be zero")
    return (((current - previous)/previous)*100)
        