import spot
import spectrl.main.compiler_utility as cutil

from inspect import signature
import buddy


class DFA():

    def __init__(self, aut, prop_num=0, pred_map={}):

        self.aut = aut
        self.prop_num = prop_num
        self.pred_map = pred_map

    def print_dfa(self):
        print(self.aut.to_str('hoa'))
        for key in self.pred_map:
            print(key, self.pred_map[key])

    def get_aut(self):
        return self.aut

    def get_prop_num(self):
        return self.prop_num

    def get_pred_map(self):
        return self.pred_map

    def add_props(self, new_props):
        last_prop = self.prop_num

        for i in range(new_props):
            self.aut.register_ap("p"+str(i+last_prop))

        self.prop_num = last_prop+new_props

    def renameprops(self, renamelist):
        bdict = spot.make_bdd_dict()
        aut_new = spot.make_twa_graph(bdict)
        aut_new.set_buchi()

        for f in self.aut.ap():
            aut_new.register_ap(f)

        existclause = buddy.bddtrue
        eqclause = buddy.bddtrue

        for key in self.pred_map.keys():

            old_prop_name = self.pred_map[key]

            if old_prop_name in renamelist.keys():
                new_prop_name = renamelist[old_prop_name]
                self.pred_map[key] = new_prop_name

                old_prop_bdd = buddy.bdd_ithvar(aut_new.register_ap(old_prop_name))
                new_prop_bdd = buddy.bdd_ithvar(aut_new.register_ap(new_prop_name))

                existclause &= old_prop_bdd
                eqclause &= buddy.bdd_biimp(old_prop_bdd, new_prop_bdd)

        # Assign states to aut_new
        state_num = self.aut.num_states()
        for i in range(state_num):
            aut_new.new_state()
        init_state = self.aut.get_init_state_number()
        aut_new.set_init_state(init_state)

        # Assign transitions with renamed propositions
        for src in range(state_num):
            for edge in self.aut.out(src):
                new_condition = buddy.bdd_exist(edge.cond & eqclause,
                                                existclause)

                aut_new.new_edge(src, edge.dst, new_condition, edge.acc)

        self.aut = aut_new

    def concatenate(self, concat_src, dfa_init, offset, accepting=True):
        # register propositions with self
        for f in dfa_init.aut.ap():
            buddy.bdd_ithvar(self.aut.register_ap(f))

        # Modify the pred_map
        pred_map_init = dfa_init.get_pred_map()
        for key in pred_map_init.keys():
            self.pred_map[key] = pred_map_init[key]

        # Add states
        num_states_init = dfa_init.aut.num_states()
        for i in range(num_states_init):
            self.aut.new_state()

        # Add transitions
        init_state_init = dfa_init.aut.get_init_state_number()
        for src in range(num_states_init):
            for edge in dfa_init.aut.out(src):
                if accepting is False:
                    self.aut.new_edge(src+offset, edge.dst+offset, edge.cond, '')
                else:
                    self.aut.new_edge(src+offset, edge.dst+offset, edge.cond, edge.acc)

                if src == init_state_init:
                    if accepting is False:
                        self.aut.new_edge(concat_src, edge.dst+offset, edge.cond, '')
                    else:
                        self.aut.new_edge(concat_src, edge.dst+offset, edge.cond, edge.acc)

        # What if concat_src is an accepting state? Should this be taken care of outside
        # not dealt with since we never accept the epsilon word

    def get_minimal_aut_from_nba(self):
        det_aut = spot.tgba_determinize(self.aut)
        min_aut = spot.minimize_wdba(det_aut)
        self.aut = min_aut


class RM():

    def __init__(self, dfa, pred_fns):

        self.num_states = 0
        self.init_state = None
        self.transition = {}
        self.reward = {}
        self.terminal = []
        self.pred_map = {}
        self.pred_fns = {}
        for p, pfunc in pred_fns.items():
            pfunc_corrected = pfunc
            if len(signature(pfunc).parameters) <= 1:
                pfunc_corrected = cutil.dummy_resource_wrapper(pfunc)
            self.pred_fns[p] = pfunc_corrected

        # private attribute. Required for Spot related functions
        self._aut = dfa.get_aut()

        self._load_from_dfa(dfa)

    def print_rm(self):

        print("Predicate map is:")
        for key in self.pred_map:
            print(key, self.pred_map[key])

        bdict = self._aut.get_dict()

        print("Initial state is {}".format(self.init_state))
        print("Accepting states are: {}".format(self.terminal))

        print("Transitions are:")
        for src in range(self.num_states):
            for dest in self.transition[src].keys():
                print("{}-->{} on {} with reward {}".format(
                    src, dest, spot.bdd_format_formula(
                        bdict, self.transition[src][dest]), self.reward[src][dest]))

        for key in self.pred_fns.keys():
            print(key, self.pred_fns[key])

    def get_initial(self):
        return self.init_state

    def step(self, rm_state, mdp_state):
        '''
        Function to compute next RM state and associated reward.
        Returns: next_state, reward
        '''

        dead_state = self.num_states*10

        if rm_state == dead_state:
            return dead_state, 0

        assignment = buddy.bddtrue
        for key in self.pred_map:
            pred = self.pred_fns[key]
            prop_name = self.pred_map[key]
            bdd_prop = buddy.bdd_ithvar(self._aut.register_ap(prop_name))
            if pred(mdp_state, None) > 0:
                assignment = buddy.bdd_and(assignment,
                                           bdd_prop)
            else:
                assignment = buddy.bdd_and(assignment,
                                           buddy.bdd_not(bdd_prop))

        for dst in self.transition[rm_state]:
            formula = self.transition[rm_state][dst]
            check_assignment = buddy.bdd_and(assignment, formula)
            # Formula AND Sat_Assignment == Sat_Assignment
            # Formula AND Not_Sat_Assignment == False
            if (not (check_assignment == buddy.bddfalse)):
                return dst, self.reward[rm_state][dst]

        return dead_state, 0

    def _load_from_dfa(self, dfa):
        aut = dfa.get_aut()
        self.num_states = aut.num_states()
        self.init_state = aut.get_init_state_number()
        self.pred_map = dfa.get_pred_map()

        for src in range(self.num_states):

            self.transition[src] = {}
            self.reward[src] = {}

            if aut.state_is_accepting(src):
                self.terminal.append(src)

            for edge in aut.out(src):

                self.transition[src][edge.dst] = edge.cond

                if (src != edge.dst) and not(aut.state_is_accepting(src)) and \
                        aut.state_is_accepting(edge.dst):
                    # Incoming edge to an accepting state from a non-accepting state
                    self.reward[src][edge.dst] = 1
                elif (src != edge.dst) and aut.state_is_accepting(src) and \
                        not(aut.state_is_accepting(edge.src)):
                    # Outgoing edge from an accepting state to a non-accepting state
                    self.reward[src][edge.dst] = -1
                else:
                    self.reward[src][edge.dst] = 0

        # check for dfa.prop_complete(). If it is not complete, then add transitions that
        # complete the reward machine even if the dfa isn't.

        if not(aut.prop_complete().is_true()):

            # print("Yallo")
            # print(aut.prop_complete())

            new_state = self.num_states
            new_state_added = False

            for src in range(self.num_states):
                condition = buddy.bddfalse
                for edge in aut.out(src):
                    condition = buddy.bdd_or(condition, edge.cond)

                if condition != buddy.bddtrue:
                    new_state_added = True
                    self.transition[src][new_state] = buddy.bdd_not(condition)

                    if aut.state_is_accepting(src):
                        self.reward[src][new_state] = -1
                    else:
                        self.reward[src][new_state] = 0

            if new_state_added:

                self.transition[new_state] = {}
                self.reward[new_state] = {}

                self.transition[new_state][new_state] = buddy.bddtrue
                self.reward[new_state][new_state] = 0

                self.num_states += 1
