# BETA LEMMANTIZER
ADJ, ADJ_SAT, ADV, NOUN, VERB = "a", "s", "r", "n", "v"
_exception_map = {}
MORPHOLOGICAL_SUBSTITUTIONS = {
        NOUN: [
            ("s", ""),
            ("ses", "s"),
            ("ves", "f"),
            ("xes", "x"),
            ("zes", "z"),
            ("ches", "ch"),
            ("shes", "sh"),
            ("men", "man"),
            ("ies", "y"),
        ],
        VERB: [
            ("s", ""),
            ("ies", "y"),
            ("es", "e"),
            ("es", ""),
            ("ed", "e"),
            ("ed", ""),
            ("ing", "e"),
            ("ing", ""),
        ],
        ADJ: [("er", ""), ("est", ""), ("er", "e"), ("est", "e")],
        ADV: [],
    }
def _morphy( form, pos, check_exceptions=False):
    # from jordanbg:
    # Given an original string x
    # 1. Apply rules once to the input to get y1, y2, y3, etc.
    # 2. Return all that are in the database
    # 3. If there are no matches, keep applying rules until you either
    #    find a match or you can't go any further
    #exceptions = _exception_map[pos]
    substitutions = MORPHOLOGICAL_SUBSTITUTIONS[pos]
    def apply_rules(forms):
        return [
            form[: -len(old)] + new
            for form in forms
            for old, new in substitutions
            if form.endswith(old)
        ]
    def filter_forms(forms):
        result = []
        seen = set()
        for form in forms:
            if form in self._lemma_pos_offset_map:
                if pos in self._lemma_pos_offset_map[form]:
                    if form not in seen:
                        result.append(form)
                        seen.add(form)
        return result
    # 0. Check the exception lists
    if check_exceptions:
        if form in exceptions:
            return filter_forms([form] + exceptions[form])
    # 1. Apply rules once to the input to get y1, y2, y3, etc.
    forms = apply_rules([form])
    # 2. Return all that are in the database (and check the original too)
    results = filter_forms([form] + forms)
    if results:
        return results
    # 3. If there are no matches, keep applying rules until we find a match
    while forms:
        forms = apply_rules(forms)
        results = filter_forms(forms)
        if results:
            return results
    # Return an empty list if we can't find anything
    return []

print(_morphy(form="Ranni's Persona: is a wandering warrior seeking the truth about the origin and destiny of the Golden Ring, a powerful relic that grants its bearers the power to create and rule worlds.",pos='v'))
              
                      