from typing import List


def information_parser(info: List[str]):
    """

    
    """
    # 
    parsed_info = []

    for line in info:
        temp = []
        raw = line.split(",")
        counter = 0
        while counter < len(raw):

            leap_steps = int(raw[counter]) * 2
            counter += 1

            q_values = {}
            fragment = raw[counter:counter + leap_steps ]
            zip_object = zip(fragment[::2], fragment[1::2])
            for key, value in zip_object:
                q_values[int(key)] = float(value)
            counter += leap_steps

            temp.append((q_values, int(raw[counter])))
            counter += 1

        parsed_info.append(temp)
    
    return parsed_info