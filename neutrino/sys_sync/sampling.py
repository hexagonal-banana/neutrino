"""Host-side capture policies; GPU selection is fixed per block-local group."""
import hashlib
import json

CONTEXT_ABI = 2
MODES = {'all': 0, 'warp': 1, 'block': 2}
_DEFAULT = object()


def validate_policy(value=_DEFAULT):
    if value is _DEFAULT:
        value = {'schema': 1, 'default': 'warp', 'rules': []}
    if not isinstance(value, dict) or set(value) - {'schema', 'default', 'rules'}:
        raise ValueError('Invalid capture policy fields')
    if type(value.get('schema')) is not int or value['schema'] != 1:
        raise ValueError('Unsupported capture policy schema')
    default, rules = value.get('default', 'warp'), value.get('rules', [])
    if not isinstance(default, str) or default not in MODES or not isinstance(rules, list):
        raise ValueError('Invalid capture policy default/rules')
    normalized = []
    for rule in rules:
        if (not isinstance(rule, dict) or set(rule) - {'kernel_glob', 'library', 'sampling'} or
                not {'kernel_glob', 'sampling'} <= set(rule)):
            raise ValueError('Invalid capture policy rule fields')
        if not isinstance(rule['sampling'], str) or rule['sampling'] not in MODES:
            raise ValueError('Unknown capture sampling mode')
        for key in ('kernel_glob', 'library'):
            if key in rule:
                text = rule[key]
                if (not isinstance(text, str) or not text or len(text) > 4096 or
                        any(not 32 <= ord(c) < 127 for c in text)):
                    raise ValueError('Capture policy patterns/labels must be printable ASCII')
        normalized.append(dict(rule))
    return {'schema': 1, 'default': default, 'rules': normalized}


def policy_json(value=_DEFAULT):
    return json.dumps(validate_policy(value), sort_keys=True, separators=(',', ':'), ensure_ascii=True)


def policy_hash(value=_DEFAULT):
    return hashlib.sha256(policy_json(value).encode()).hexdigest()


def glob_match(pattern, text):
    """Full-symbol matching: only '*' and '?' are special; brackets are literal."""
    p = t = 0
    star, retry = -1, 0
    while t < len(text):
        if p < len(pattern) and pattern[p] == '*':
            star, retry = p, t
            p += 1
        elif p < len(pattern) and (pattern[p] == '?' or pattern[p] == text[t]):
            p += 1
            t += 1
        elif star >= 0:
            retry += 1
            t, p = retry, star + 1
        else:
            return False
    return all(c == '*' for c in pattern[p:])


def resolve_policy(value, kernel, library):
    policy = validate_policy(value)
    for index, rule in enumerate(policy['rules']):
        if ('library' not in rule or rule['library'] == library) and glob_match(rule['kernel_glob'], kernel):
            return rule['sampling'], index
    return policy['default'], -1
