import os 
import json
from glob import glob
from utils import get_code, SexpCache, set_paths, extract_code, dst_filename
from serapi import SerAPI
from time import time
import sexpdata
from proof_tree import ProofTree
import pdb
import pickle


def check_topology(proof_steps):
    'Only the first focused goal can be decomposed. This facilitates reconstructing the proof tree'
    # the proof starts with one focused goal
    if not (len(proof_steps[0]['goal_ids']['fg']) == 1 and proof_steps[0]['goal_ids']['bg'] == []):
        return False

    for i, step in enumerate(proof_steps[:-1]):
        step = step['goal_ids']
        next_step = proof_steps[i + 1]['goal_ids']
        if next_step['fg'] == step['fg'] and next_step['bg'] == step['bg']:  # all goals remain the same
            pass
        elif next_step['fg'] == step['fg'][1:] and next_step['bg'] == step['bg']:  # a focused goal is solved
            pass
        elif len(step['fg']) == 1 and next_step['fg'] == step['bg'] and next_step['bg'] == []:
            pass
        elif step['fg'] != [] and next_step['fg'] == [step['fg'][0]] and next_step['bg'] == step['fg'][1:] + step['bg']:  # zoom in
            pass
        elif step['fg'] == [] and next_step['fg'] == [step['bg'][0]] and next_step['bg'] == step['bg'][1:]:  # zoom out
            pass
        elif step['fg'] != [] and ''.join([str(x) for x in next_step['fg']]).endswith(''.join([str(x) for x in step['fg'][1:]])) and \
             step['fg'][0] not in next_step['fg'] and next_step['bg'] == step['bg']:  # a focused goal is decomposed
            pass
        else:
            return False

    return True


def goal_is_prop(goal, serapi):
    'Check if the sort of a goal is Prop'
    sort = serapi.query_type(goal['sexp'], return_str=True)
    return sort == 'Prop'


def record_proof(num_extra_cmds, line_nb, script, sexp_cache, serapi, args, proof_dict):
    proof_data = {
        'line_nb': num_extra_cmds + line_nb,
        'env': {},
        'steps': [],
        'goals': {},
        'proof_tree': None,
    }

    # get the global environment
    serapi.set_timeout(3600)
    constants, inductives = serapi.query_env(args.file[:-5] + '.vo')
    serapi.set_timeout(args.timeout)

    # execute the proof
    for num_executed, (code_line, tags) in enumerate(script, start=line_nb + 1):
        if 'END_TACTIC' in tags:
            return None
        assert tags['VERNAC_TYPE'] != 'VernacProof'
        if tags['VERNAC_TYPE'] not in ['VernacEndProof', 'VernacBullet', 'VernacSubproof', 'VernacEndSubproof', 'VernacExtend']:
            return None

        # keep track of the goals
        fg_goals, bg_goals, shelved_goals, given_up_goals = serapi.query_goals()
        if shelved_goals + given_up_goals != []:
            return None
        if num_executed == 0:  # we only consider Prop
            assert fg_goals != []
            if not goal_is_prop(fg_goals[0], serapi):
                return None
        responses, _ , msg_str = serapi.execute("Show Proof.")
        proof_dict[code_line[:-1]] = msg_str
        # run the tactic
        if args.debug:
            print('PROOF %d: %s' % (num_executed, code_line))
        serapi.execute(code_line)

        # the proof ends
        if tags['VERNAC_TYPE'] == 'VernacEndProof':
            break

    return proof_dict
    

def get_proof(sexp_cache, args):
    coq_filename = os.path.splitext(args.file)[0] + '.v'
    fields = coq_filename.split(os.path.sep)
    loc2code = get_code(open(coq_filename, 'rb').read())
    meta = open(args.file).read()
    coq_code = extract_code(meta, loc2code)
    file_data = json.load(open(os.path.join(args.data_path, args.file[13:-5] + '.json')))

    gal_data_dict = {}

    with SerAPI(args.timeout, args.debug) as serapi:
        num_extra_cmds = len(set_paths(meta, serapi, sexp_cache))
        
        # process the coq code
        proof_start_lines = []
        in_proof = False
        proof_dict = {}
        for num_executed, (code_line, tags) in enumerate(coq_code):
            # assert code_line == file_data['vernac_cmds'][num_extra_cmds + num_executed][0]
            if 'PROOF_NAME' in tags:  # the proof ends
                serapi.pop() 
                args.proof = tags['PROOF_NAME']
                line_nb = proof_start_lines[-1]
                new_proof_dict = record_proof(num_extra_cmds, line_nb, coq_code[line_nb + 1:], sexp_cache, serapi, args, proof_dict)
                if new_proof_dict is not None:
                    gal_data_dict[args.proof] = new_proof_dict
                else:
                    gal_data_dict[args.proof] = proof_dict
                proof_dict = {}
                in_proof = False
                continue
            # execute the code
            if args.debug:
                print('%d: %s' % (num_executed, code_line))
            if in_proof == True:
                # print("CODE LINE MID: ")
                # print(code_line)
                responses, _ , msg_str = serapi.execute("Show Proof.")
                # print("MSG STR MID:")
                # print(msg_str)
                proof_dict[code_line[:-1]] = msg_str
            serapi.execute(code_line)
            if serapi.has_open_goals():
                if not in_proof:  # the proof starts
                    in_proof = True
                    proof_start_lines.append(num_executed)
                    serapi.push()
            else:
                in_proof = False

    return gal_data_dict 


def dump(proof_data, args):
    dirname = dst_filename(args.file, args.data_path) + '-PROOFS'
    json.dump(proof_data, open(os.path.join(dirname, args.proof + '.json'), 'wt'))
  
  
if __name__ == '__main__':
    import sys
    sys.setrecursionlimit(100000)
    import argparse
    arg_parser = argparse.ArgumentParser(description='Extract the proofs from Coq source code')
    arg_parser.add_argument('--debug', action='store_true')
    arg_parser.add_argument('--file', type=str, help='The meta file to process')
    arg_parser.add_argument('--proof', type=str, default='all_proofs')
    arg_parser.add_argument('--timeout', type=int, default=600, help='Timeout for SerAPI')
    arg_parser.add_argument('--data_path', type=str, default='./data')
    arg_parser.add_argument('--other_path', type=str, default='./gal_data')
    args = arg_parser.parse_args()
    print(args)

    dirname = dst_filename(args.file, args.data_path) + '-PROOFS'
    try:
        os.makedirs(dirname)
    except os.error:
         pass
    db_path = os.path.join(dirname, args.proof + '-sexp_cache')
    sexp_cache = SexpCache(db_path)

    
    gal_data_dict = get_proof(sexp_cache, args)
    f = open(os.path.join(args.data_path, args.file[13:-5] + '.p'), "wb")
    pickle.dump(gal_data_dict,f)
    f.close()