'''
This file contains functions that will be utilized by both part1 and part2.
'''

from translator import UniversalTranslator


evaluated_settings = []
decode_rates = []

def decode_rate(s: str) -> float:
    """
    :param s: a string representing the attempted message decoding
    :return: fraction of words successfully decoded, expressed as a value
             between 0 and 1 (inclusive)
    """
    tokens = s.split()
    num_tokens = len(tokens)
    num_decoded = sum(1 for token in tokens if not token.isdigit())

    return (num_decoded / num_tokens)

def objective(settings: list) -> float:
    """
    the objective function to minimize

    :param settings: the knob settings to be tried
    :return: the fraction of words successfully decoded
    """
    translator = UniversalTranslator(n_dim=len(settings))
    translated_string = translator.translate(settings)

    score = decode_rate(translated_string)

    if len(settings) == 2:
        evaluated_settings.append(settings.copy())
        decode_rates.append(score)
    
    return -score
