from spectrl.main.monitor import Monitor_Automaton
from spectrl.main.reward_machine import DFA, RM
from inspect import signature
import spectrl.main.compiler_utility as cutil


import enum
import numpy as np

import spot
import buddy


MAX_PRED_VAL = 1000000.


class Cons(enum.Enum):
    '''
    Constructors for TaskSpec
    '''
    ev = 1
    alw = 2
    seq = 3
    choose = 4
    ite = 5


class TaskSpec:
    '''
    Specification AST.

    Fields:
        cons: int (Has to refer to a valid constructor in Cons)
        predicate: np.array(state_dim), np.array(resource_dim) -> float
        subtasks: [TaskSpec]

    Some functions in this class assume presence of subtasks as needed,
    according to syntax of language in the paper.
    Predicate can be str (used for symbolic representation)
    '''

    def __init__(self, cons, predicate, subtasks):
        self.cons = cons
        self.predicate = predicate
        self.subtasks = subtasks

    def quantitative_semantics_fast(self, traj, sys_dim, use_prefix=False):
        sys_traj = [state[:sys_dim] for state in traj]
        res_traj = [state[sys_dim:] for state in traj]
        if use_prefix:
            return self.quantitative_semantics_dp_fast(sys_traj, res_traj)[1][len(traj)-1]
        else:
            return self.quantitative_semantics_dp_fast(sys_traj, res_traj)[0][0]

    def quantitative_semantics_dp_fast(self, sys_traj, res_traj):
        '''
        Does not support conditional statements
        Returns quantitative semnatics for all suffixes (retval[0]) and prefixes (retval[1])
        suffixes not available once a sequence operator is encountered with first task != ev
        prefixes not available once a sequence operator is encountered with second task != ev
        '''
        n = len(sys_traj)
        retval = np.zeros((2, n))

        # atomic task
        if self.cons == Cons.ev:
            for i in range(n):
                if i == 0:
                    retval[0][n-1] = self.predicate(sys_traj[n-1], res_traj[n-1])
                    retval[1][0] = self.predicate(sys_traj[0], res_traj[0])
                else:
                    retval[0][n-i-1] = max(retval[0][n-i], self.predicate(sys_traj[n-i-1],
                                                                          res_traj[n-i-1]))
                    retval[1][i] = max(retval[1][i-1], self.predicate(sys_traj[i],
                                                                      res_traj[i]))
            return retval

        # always constraint
        # this function also supports always formala without a subtask
        if self.cons == Cons.alw:
            subval = np.array([[MAX_PRED_VAL]*n]*2)
            if self.subtasks[0] is not None:
                subval = self.subtasks[0].quantitative_semantics_dp(sys_traj, res_traj)
            for i in range(n):
                if i == 0:
                    retval[0][n-1] = self.predicate(sys_traj[n-1], res_traj[n-1])
                    retval[1][0] = self.predicate(sys_traj[0], res_traj[0])
                else:
                    retval[0][n-i-1] = min(retval[0][n-i], self.predicate(sys_traj[n-i-1],
                                                                          res_traj[n-i-1]))
                    retval[1][i] = min(retval[1][i-1], self.predicate(sys_traj[i],
                                                                      res_traj[i]))
            for i in range(n):
                retval[0][i] = min(retval[0][i], subval[0][i])
                retval[1][i] = min(retval[1][i], subval[1][i])
            return retval

        # sequence
        if self.cons == Cons.seq:
            if self.subtasks[0].cons == Cons.ev:
                subval = self.subtasks[1].quantitative_semantics_dp(sys_traj, res_traj)
                for i in range(n):
                    if i == 0:
                        retval[0][n-1] = -MAX_PRED_VAL
                    else:
                        retval[0][n-i-1] = max(retval[0][n-i],
                                               min(self.subtasks[0].predicate(sys_traj[n-i-1],
                                                                              res_traj[n-i-1]),
                                                   subval[0][n-i]))
                        subval[0][n-i-1] = max(subval[0][n-i], subval[0][n-i-1])
            if self.subtasks[1].cons == Cons.ev:
                subval = self.subtasks[0].quantitative_semantics_dp(sys_traj, res_traj)
                for i in range(n):
                    if i == 0:
                        retval[1][0] = -MAX_PRED_VAL
                    else:
                        retval[1][i] = max(retval[1][i-1],
                                           min(self.subtasks[1].predicate(sys_traj[i],
                                                                          res_traj[i]),
                                               subval[1][i-1]))
                        subval[1][i] = max(subval[1][i-1], subval[1][i])
            return retval

        # choice
        if self.cons == Cons.choose:
            subval1 = self.subtasks[0].quantitative_semantics_dp(sys_traj, res_traj)
            subval2 = self.subtasks[1].quantitative_semantics_dp(sys_traj, res_traj)
            for i in range(n):
                retval[0][i] = max(subval1[0][i], subval2[0][i])
                retval[1][i] = max(subval1[1][i], subval2[1][i])
            return retval

    def quantitative_semantics(self, traj, sys_dim):
        sys_traj = [state[:sys_dim] for state in traj]
        res_traj = [state[sys_dim:] for state in traj]
        return self.quantitative_semantics_dp(sys_traj, res_traj)[0][len(traj)-1]

    def quantitative_semantics_dp(self, sys_traj, res_traj):
        '''
        Does not support conditional statements
        Returns quantitative semnatics for all intervals
        '''
        n = len(sys_traj)
        retval = -MAX_PRED_VAL * np.ones((n, n))

        # atomic task
        if self.cons == Cons.ev:
            for i in range(n):
                for j in range(i, n):
                    if j == i:
                        retval[i][j] = self.predicate(sys_traj[i], res_traj[i])
                    else:
                        retval[i][j] = max(retval[i][j-1], self.predicate(sys_traj[j], res_traj[j]))
            return retval

        # always constraint
        # this function also supports always formala without a subtask
        if self.cons == Cons.alw:
            subval = MAX_PRED_VAL * np.ones((n, n))
            if self.subtasks[0] is not None:
                subval = self.subtasks[0].quantitative_semantics_dp(sys_traj, res_traj)
            for i in range(n):
                for j in range(i, n):
                    if j == i:
                        retval[i][j] = self.predicate(sys_traj[i], res_traj[i])
                    else:
                        retval[i][j] = min(retval[i][j-1], self.predicate(sys_traj[j], res_traj[j]))
            for i in range(n):
                for j in range(i, n):
                    retval[i][j] = min(retval[i][j], subval[i][j])
            return retval

        # sequence
        if self.cons == Cons.seq:
            subval1 = self.subtasks[0].quantitative_semantics_dp(sys_traj, res_traj)
            subval2 = self.subtasks[1].quantitative_semantics_dp(sys_traj, res_traj)
            for i in range(n):
                for j in range(i, n):
                    retval[i][j] = -MAX_PRED_VAL
                    for k in range(i, j):
                        retval[i][j] = max(retval[i][j], min(subval1[i][k], subval2[k+1][j]))
            return retval

        # choice
        if self.cons == Cons.choose:
            subval1 = self.subtasks[0].quantitative_semantics_dp(sys_traj, res_traj)
            subval2 = self.subtasks[1].quantitative_semantics_dp(sys_traj, res_traj)
            for i in range(n):
                for j in range(i, n):
                    retval[i][j] = max(subval1[i][j], subval2[i][j])
            return retval

    def get_monitor(self, state_dim, res_dim, local_reward_bound):
        '''
        Returns a monitor: Monitor_Automaton.
        '''

        # atomic evetually task
        if self.cons == Cons.ev:

            mpred = cutil.monitor_predicate(self.predicate, state_dim)

            def pred_update(state, reg):
                return np.array([mpred(state, reg)[1]])

            def frew(state, reg):
                return reg[0]

            e00 = (0, cutil.true_predicate(
                local_reward_bound), cutil.id_update)
            e01 = (1, mpred, pred_update)
            e11 = (1, cutil.true_predicate(
                local_reward_bound), cutil.id_update)
            transitions = [[e00, e01], [e11]]
            rewards = [None, frew]

            return Monitor_Automaton(2, 1, state_dim + res_dim, np.array([0.0]), transitions,
                                     rewards)

        # adding safety constraints
        elif self.cons == Cons.alw:

            # construct monitor for sub-formula
            mon = self.subtasks[0].get_monitor(
                state_dim, res_dim, local_reward_bound)

            # add saftey constraint
            mpred = cutil.monitor_predicate(self.predicate, state_dim)

            n_states = mon.n_states
            n_registers = mon.n_registers + 1
            input_dim = mon.input_dim
            init_reg = np.concatenate(
                [mon.init_registers, np.array([local_reward_bound])])

            transitions = []
            for ve in mon.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q,
                                   cutil.project_predicate(
                                       p, 0, mon.n_registers),
                                   cutil.alw_update(u, mpred)))
                transitions.append(ve_new)

            rewards = []
            for rew in mon.rewards:
                if rew is None:
                    rewards.append(None)
                else:
                    rewards.append(cutil.alw_reward(rew))

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards)

        # sequence
        elif self.cons == Cons.seq:

            # construct monitors for subformulas
            mon1 = self.subtasks[0].get_monitor(
                state_dim, res_dim, local_reward_bound)
            mon2 = self.subtasks[1].get_monitor(
                state_dim, res_dim, local_reward_bound)

            # construct monitor for sequence
            n_states = mon1.n_states + mon2.n_states
            n_registers = max(mon1.n_registers, mon2.n_registers + 1)
            input_dim = mon1.input_dim
            init_reg = np.zeros(n_registers)
            init_reg[:mon1.n_registers] = mon1.init_registers

            transitions = []
            for qu in range(mon1.n_states):
                ve = mon1.transitions[qu]
                ve_new = []

                # Delta1
                for (qv, p, u) in ve:
                    ve_new.append((qv,
                                   cutil.project_predicate(
                                       p, 0, mon1.n_registers),
                                   cutil.project_update(u, 0, mon1.n_registers)))

                # Delta1->2
                if mon1.rewards[qu] is not None:
                    for (qv, p, u) in mon2.transitions[0]:
                        q2 = qv + mon1.n_states
                        p2 = cutil.rew_pred(
                            p, mon1.rewards[qu], mon2.init_registers, 0, mon1.n_registers)
                        u2 = cutil.seq_update(n_registers, mon1.n_registers, mon2.n_registers,
                                              mon2.init_registers, mon1.rewards[qu], u)
                        ve_new.append((q2, p2, u2))

                transitions.append(ve_new)

            for ve in mon2.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q + mon1.n_states,
                                   cutil.project_predicate(
                                       p, 0, mon2.n_registers),
                                   cutil.project_update(u, 0, mon2.n_registers)))
                transitions.append(ve_new)

            rewards = [None]*mon1.n_states
            for rew in mon2.rewards:
                if rew is not None:
                    rewards.append(cutil.seq_reward(rew, mon2.n_registers))
                else:
                    rewards.append(None)

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards)

        # choice
        elif self.cons == Cons.choose:

            # Construct monitors for subformulas
            mon1 = self.subtasks[0].get_monitor(
                state_dim, res_dim, local_reward_bound)
            mon2 = self.subtasks[1].get_monitor(
                state_dim, res_dim, local_reward_bound)

            # combine
            # initial state is merged, state numbers of first monitor do not change
            n_states = mon1.n_states + mon2.n_states - 1
            n_registers = mon1.n_registers + mon2.n_registers
            input_dim = mon1.input_dim
            init_reg = np.concatenate(
                [mon1.init_registers, mon2.init_registers])

            transitions = []

            # Delta0
            # us[0] stores loop update for the first monitor, us[1] for second monitor
            us = []
            # Set of transitions from initial state: None is used as a placeholder for self loop
            trans_init = [None]
            for (q1, p1, u1) in mon1.transitions[0]:
                if q1 == 0:
                    us.append(u1)
                else:
                    trans_init.append((q1,
                                       cutil.project_predicate(
                                           p1, 0, mon1.n_registers),
                                       cutil.project_update(u1, 0, mon1.n_registers, clean=True)))

            for (q2, p2, u2) in mon2.transitions[0]:
                if q2 == 0:
                    us.append(u2)
                else:
                    trans_init.append((q2 + mon1.n_states - 1,
                                       cutil.project_predicate(
                                           p2, mon1.n_registers, n_registers),
                                       cutil.project_update(u2, mon1.n_registers, n_registers,
                                                            clean=True)))

            def loop_update(state, reg):
                return np.concatenate([us[0](state, reg[0:mon1.n_registers]),
                                       us[1](state, reg[mon1.n_registers:n_registers])])
            trans_init[0] = (0, cutil.true_predicate(
                local_reward_bound), loop_update)
            transitions.append(trans_init)

            # Delta1: Add all transitions in monitor 1
            for ve in mon1.transitions[1:]:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q,
                                   cutil.project_predicate(
                                       p, 0, mon1.n_registers),
                                   cutil.project_update(u, 0, mon1.n_registers)))
                transitions.append(ve_new)

            # Delta2: Add all transitions in monitor 2
            for ve in mon2.transitions[1:]:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q + mon1.n_states - 1,
                                   cutil.project_predicate(
                                       p, mon1.n_registers, n_registers),
                                   cutil.project_update(u, mon1.n_registers, n_registers)))
                transitions.append(ve_new)

            rewards = [None]
            for rew in mon1.rewards[1:]:
                if rew is not None:
                    rewards.append(cutil.project_reward(
                        rew, 0, mon1.n_registers))
                else:
                    rewards.append(None)

            for rew in mon2.rewards[1:]:
                if rew is not None:
                    rewards.append(cutil.project_reward(
                        rew, mon1.n_registers, n_registers))
                else:
                    rewards.append(None)

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards)

        # conditional
        else:

            # Construct monitors for sub formulas
            mon1 = self.subtasks[0].get_monitor(
                state_dim, res_dim, local_reward_bound)
            mon2 = self.subtasks[1].get_monitor(
                state_dim, res_dim, local_reward_bound)

            b = cutil.monitor_predicate(self.predicate, state_dim)
            notb = cutil.neg_pred(b)

            # Combine monitors
            n_states = mon1.n_states + mon2.n_states + 1
            n_registers = max(mon1.n_registers, mon2.n_registers)
            input_dim = mon1.input_dim
            init_reg = np.zeros(n_registers)

            transitions = []

            # Delta0
            trans_init = []

            for (q, p, u) in mon1.transitions[0]:
                trans_init.append((q+1,
                                   cutil.conj_pred(p, b, mon1.init_registers),
                                   cutil.init_update(u, mon1.init_registers)))
            for (q, p, u) in mon2.transitions[0]:
                trans_init.append((q + mon1.n_states + 1,
                                   cutil.conj_pred(
                                       p, notb, mon2.init_registers),
                                   cutil.init_update(u, mon2.init_registers)))
            transitions.append(trans_init)

            # Delta1

            for ve in mon1.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q+1,
                                   cutil.project_predicate(
                                       p, 0, mon1.n_registers),
                                   cutil.project_update(u, 0, mon1.n_registers)))
                transitions.append(ve_new)

            # Delta2

            for ve in mon2.transitions:
                ve_new = []
                for (q, p, u) in ve:
                    ve_new.append((q + mon1.n_states + 1,
                                   cutil.project_predicate(
                                       p, 0, mon2.n_registers),
                                   cutil.project_update(u, 0, mon2.n_registers)))
                transitions.append(ve_new)

            rewards = [None]

            for rew in mon1.rewards:
                if rew is None:
                    rewards.append(None)
                else:
                    rewards.append(cutil.project_reward(
                        rew, 0, mon1.n_registers))

            for rew in mon2.rewards:
                if rew is None:
                    rewards.append(None)
                else:
                    rewards.append(cutil.project_reward(
                        rew, 0, mon2.n_registers))

            return Monitor_Automaton(n_states, n_registers, input_dim, init_reg, transitions,
                                     rewards)

    def get_dfa(self):

        if self.cons == Cons.ev:
            bdict = spot.make_bdd_dict()
            aut = spot.translate('F p0', 'BA', 'Deterministic', dict=bdict)
            pred_map = {}
            pred_map[self.predicate] = 'p0'
            return DFA(aut, 1, pred_map)

        if self.cons == Cons.alw:

            # DFA for subtask
            dfa_temp = self.subtasks[0].get_dfa()
            state_num = dfa_temp.aut.num_states()
            prop_num = dfa_temp.get_prop_num()
            aut_temp = dfa_temp.get_aut()
            pred_map_temp = dfa_temp.get_pred_map()

            bdict = spot.make_bdd_dict()
            aut = spot.make_twa_graph(bdict)
            aut.set_buchi()

            for f in aut_temp.ap():
                buddy.bdd_ithvar(aut.register_ap(f))

            if (self.predicate in pred_map_temp.keys()):
                new_prop = pred_map_temp[self.predicate]
            else:
                new_prop_sym = "p"+str(prop_num)
                pred_map_temp[self.predicate] = new_prop_sym
                new_prop = buddy.bdd_ithvar(aut.register_ap(new_prop_sym))

            # Assign states
            for i in range(state_num):
                aut.new_state()
            init_state = aut_temp.get_init_state_number()
            aut.set_init_state(init_state)

            # Assign transitions
            for src in range(state_num):
                for edge in aut_temp.out(src):
                    aut.new_edge(src, edge.dst, buddy.bdd_and(edge.cond, new_prop), edge.acc)

            # For completeness, add a non-accepting sink state

            # Issue with this is that it makes the accepting condition transition-based
            # aut.new_state()
            # for src in range(state_num):
                # aut.new_edge(src, state_num, bdd_not(new_prop), '')
            # aut.new_edge(state_num, state_num, bddtrue, '')

            aut = spot.tgba_determinize(aut)
            aut = spot.minimize_wdba(aut)

            # d = DFA(aut, prop_num+1, pred_map_temp)

            # d.print_dfa()

            return DFA(aut, prop_num+1, pred_map_temp)

        if self.cons == Cons.choose:

            dfa1 = self.subtasks[0].get_dfa()
            state_num1 = dfa1.aut.num_states()
            prop_num1 = dfa1.get_prop_num()
            _ = dfa1.get_aut()
            pred_map1 = dfa1.get_pred_map()

            dfa2 = self.subtasks[1].get_dfa()
            prop_num2_init = dfa2.get_prop_num()

            dfa2.add_props(prop_num1)

            # Rename every initial prop in dfa2 so that pi becomes p(i+num_prop1)
            renamelist = {}
            for i in range(prop_num2_init):
                old_prop_name = "p"+str(i)
                new_prop_name = "p" + str(i+prop_num1)
                renamelist[old_prop_name] = new_prop_name
            dfa2.renameprops(renamelist)

            # Rename props in dfa2 again so that predicates that are common to dfa1 and dfa2
            # have the same props.
            renamelist = {}
            pred_map2 = dfa2.get_pred_map()
            for pred in pred_map1.keys():
                if pred in pred_map2.keys():
                    old_prop = pred_map2[pred]
                    new_prop_name = pred_map1[pred]
                    renamelist[old_prop] = new_prop_name
            if len(renamelist) != 0:
                dfa2.renameprops(renamelist)

            # CREATE NEW DFA
            bdict = spot.make_bdd_dict()
            aut = spot.make_twa_graph(bdict)
            aut.set_buchi()

            # Add initial state to aut
            aut.new_state()
            init_state_num = 0
            aut.set_init_state(init_state_num)

            # Create the DFA
            dfa = DFA(aut, prop_num1+prop_num2_init, {})
            dfa.concatenate(init_state_num, dfa1, 1)
            dfa.concatenate(init_state_num, dfa2, state_num1+1)

            dfa.get_minimal_aut_from_nba()

            return dfa

        if self.cons == Cons.seq:

            dfa1 = self.subtasks[0].get_dfa()
            state_num1 = dfa1.aut.num_states()
            prop_num1 = dfa1.get_prop_num()
            _ = dfa1.get_aut()
            pred_map1 = dfa1.get_pred_map()

            dfa2 = self.subtasks[1].get_dfa()
            prop_num2_init = dfa2.get_prop_num()
            dfa2.add_props(prop_num1)

            # Rename every initial prop in dfa2 so that pi becomes p(i+num_prop1)
            renamelist = {}
            for i in range(prop_num2_init):
                old_prop_name = "p"+str(i)
                new_prop_name = "p" + str(i+prop_num1)
                renamelist[old_prop_name] = new_prop_name
            dfa2.renameprops(renamelist)

            # Rename props in dfa2 again so that predicates that are common to dfa1 and dfa2 have
            # the same props.
            renamelist = {}
            pred_map2 = dfa2.get_pred_map()
            for pred in pred_map1.keys():
                if pred in pred_map2.keys():
                    old_prop = pred_map2[pred]
                    new_prop_name = pred_map1[pred]
                    renamelist[old_prop] = new_prop_name
            if len(renamelist) != 0:
                dfa2.renameprops(renamelist)

            # CREATE NEW AUT
            bdict = spot.make_bdd_dict()
            aut = spot.make_twa_graph(bdict)
            aut.set_buchi()

            # Add initial state to aut
            aut.new_state()
            init_state_num = 0
            aut.set_init_state(init_state_num)

            # CREATE NEW DFA
            dfa = DFA(aut, prop_num1+prop_num2_init, pred_map1)

            offset = 1
            dfa.concatenate(init_state_num, dfa1, offset, False)

            num_accepting_state = 0
            num_state_dfa1 = dfa1.aut.num_states()
            for s in range(num_state_dfa1):
                if dfa1.aut.state_is_accepting(s):

                    new_acc_num = s + offset
                    offset_from_acc = num_state_dfa1*(num_accepting_state+1) + offset
                    dfa.concatenate(new_acc_num, dfa2, offset_from_acc)
                    num_accepting_state += 1

            dfa.get_minimal_aut_from_nba()

            return dfa

    def get_rm(self, pred_fns):
        return RM(self.get_dfa(), pred_fns)

    def add_pred_fns(self, pred_fns):
        '''
        Convert symbolic spec into interpretable spec.
        pred_fns: Dict[str, Callable]
        '''
        if self.cons == Cons.ev:
            return ev(pred_fns[self.predicate])
        elif self.cons == Cons.alw:
            phi1 = self.subtasks[0].add_pred_fns(pred_fns)
            return alw(pred_fns[self.predicate], phi1)
        elif self.cons == Cons.seq:
            phi1 = self.subtasks[0].add_pred_fns(pred_fns)
            phi2 = self.subtasks[1].add_pred_fns(pred_fns)
            return seq(phi1, phi2)
        elif self.cons == Cons.choose:
            phi1 = self.subtasks[0].add_pred_fns(pred_fns)
            phi2 = self.subtasks[1].add_pred_fns(pred_fns)
            return choose(phi1, phi2)
        elif self.cons == Cons.ite:
            phi1 = self.subtasks[0].add_pred_fns(pred_fns)
            phi2 = self.subtasks[1].add_pred_fns(pred_fns)
            return ite(pred_fns[self.predicate], phi1, phi2)

# ============================================================================================
# API for building Task Specifications


def lor(p1, p2):
    '''
    Logical OR
    p1, p2 : np.array([state_dim]), np.array([res_dim]) -> Float
    '''
    def p(sys_state, res_state):
        return max(p1(sys_state, res_state), p2(sys_state, res_state))
    return p


def land(p1, p2):
    '''
    Logical AND
    p1, p2 : np.array([state_dim]), np.array([res_dim]) -> Float
    '''
    def p(sys_state, res_state):
        return min(p1(sys_state, res_state), p2(sys_state, res_state))
    return p


def ev(p):
    '''
    Atomic eventually task
    p : np.array([state_dim]), np.array([res_dim]) -> Float OR
        np.array([state_dim]) -> Float
        Int
    '''
    if not callable(p) or len(signature(p).parameters) > 1:
        return TaskSpec(Cons.ev, p, [])
    else:
        return TaskSpec(Cons.ev, cutil.dummy_resource_wrapper(p), [])


def alw(p, phi):
    '''
    Safety constraints
    p : np.array([state_dim]) * np.array([res_dim]) -> Float OR
        np.array([state_dim]) -> Float
        Int
    phi : TaskSpec
    '''
    if not callable(p) or len(signature(p).parameters) > 1:
        return TaskSpec(Cons.alw, p, [phi])
    else:
        return TaskSpec(Cons.alw, cutil.dummy_resource_wrapper(p), [phi])


def seq(phi1, phi2):
    '''
    Sequence of tasks
    phi1, phi2 : TaskSpec
    '''
    return TaskSpec(Cons.seq, None, [phi1, phi2])


def choose(phi1, phi2):
    '''
    Choice of tasks
    phi1, phi2 : TaskSpec
    '''
    return TaskSpec(Cons.choose, None, [phi1, phi2])


def ite(p, phi1, phi2):
    '''
    Conditional
    p : np.array([state_dim]) * np.array([res_dim]) -> Float OR
        np.array([state_dim]) -> Float
        Int
    phi1, phi2 : TaskSpec
    '''
    if not callable(p) or len(signature(p).parameters) > 1:
        return TaskSpec(Cons.ite, p, [phi1, phi2])
    else:
        return TaskSpec(Cons.ite, cutil.dummy_resource_wrapper(p), [phi1, phi2])
