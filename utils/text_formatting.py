def format_nums_to_string(numbers_list):
    """
    Format a list of numbers into a string, for display purposes.

    Every element is separated with a comma, except for the last one.

    :param numbers_list: List with numbers.
    :type numbers_list: list
    :return: String with the list's elements separated by a comma.
    :rtype: string
    """
    txt = f""
    for member_id in range(len(numbers_list)):
        txt += f"{numbers_list[member_id]}, " if member_id < len(numbers_list)-1 else f"{numbers_list[member_id]}"
    return txt