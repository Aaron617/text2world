import re
import json
import os

def extract_domain_name(pddl_text):
    pattern = r"\(define \(domain\s+(\S+)"
    match = re.search(pattern, pddl_text)
    if match:
        return match.group(1).replace(')', '')
    else:
        return None

def extract_pddl(text):
    if not isinstance(text, str):
        raise TypeError("Input 'text' must be a string")
    if not text.strip():
        raise ValueError("Input 'text' cannot be empty")
        
    pattern = r"```pddl\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches == []:
        pattern = r"```\n(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL)
    
    if not matches:
        raise ValueError("No PDDL code block found in the text")
        
    pddl = max(matches, key=len).replace('```pddl', '').replace('```', '').strip()
    if not pddl:
        raise ValueError("Extracted PDDL code is empty")
        
    return pddl

from tarski.io import PDDLReader
from tarski.syntax.formulas import *
import traceback
def _checker(_domain, raise_on_error=True):
    try:
        reader = PDDLReader(raise_on_error=raise_on_error)

        reader.parse_domain_string(_domain)
        
        return True, 'Success'
        
    except Exception as e:
        exception_type = type(e).__name__
        traceback_info = traceback.format_exc()
        error_message = f"{exception_type}: {str(e)}"
        return False, error_message

def parse_actions(pddl_domain):
    """Parse domain actions and return a map of action names to parameter counts"""
    # Clean up the domain string
    pddl_domain = pddl_domain.strip()
        
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain_string(pddl_domain)
        
    return reader.problem.actions

def parse_predicates(pddl_domain):
    """Parse domain predicates and return a map of predicate names to arities"""
    pddl_domain = pddl_domain.strip()
    
    reader = PDDLReader(raise_on_error=True)
    reader.parse_domain_string(pddl_domain)
    predicate_map = {}
    for pred in reader.problem.language.predicates:
        if str(pred.symbol) not in ['=', '!=']:
            predicate_map[str(pred.symbol)] = pred.arity
    return predicate_map

def _purge_comments(pddl_str):
    # Purge comments from the given string
    while True:
        match = re.search(r";(.*)\n", pddl_str)
        if match is None:
            break  # Changed from return to break to handle newlines after
        start, end = match.start(), match.end()
        pddl_str = pddl_str[:start]+pddl_str[end-1:]
    
    # First remove empty lines that only contain whitespace
    pddl_str = re.sub(r'\n\s+\n', '\n\n', pddl_str)
    # Then remove consecutive newlines (more than 2) with just 2 newlines
    pddl_str = re.sub(r'\n{2,}', '\n\n', pddl_str)
    
    return pddl_str

def pddl_tokenize(text):
    text = _purge_comments(text)
    
    pddl_patterns = [
        r'\(\s*define',
        r':domain',
        r':requirements',
        r':types',
        r':predicates',
        r':action',
        r':parameters',
        r':precondition',
        r':effect',
        
        r':constants',
        r':functions',
        r':durative-action',
        r':derived',
        
        r':strips',
        r':typing',
        r':negative-preconditions',
        r':disjunctive-preconditions',
        r':equality',
        r':existential-preconditions',
        r':universal-preconditions',
        r':quantified-preconditions',
        r':conditional-effects',
        r':fluents',
        r':adl',
        r':durative-actions',
        r':derived-predicates',
        
        r'not',
        r'and',
        r'or',
        r'exists',
        r'forall',
        r'when',
        r'imply',
        r'preference',
        
        r'increase',
        r'decrease',
        r'assign',
        r'scale-up',
        r'scale-down',
        
        r'[<>=]=?', 
        
        r'-?\d+\.?\d*',   
        r'#t',           
        r'\?duration',  
        
        r'\?[a-zA-Z][a-zA-Z0-9_-]*',  
        r'[a-zA-Z][a-zA-Z0-9_-]*',    
        r'[!$%&*+./<=>?@^_~-][!$%&*+./<=>?@^_~-]*',
        
        r'\(|\)',
        r'-',
    ]
    
    pattern = '|'.join(pddl_patterns)
    tokens = re.findall(pattern, text, re.IGNORECASE)
    
    tokens = [t.strip().lower() for t in tokens if t.strip()]
    return tokens

def test_extract_pddl():
    gpt_response = ""
    pddl_domain = extract_pddl(gpt_response)
    print(pddl_domain)
