import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pprint
import pickle


pp = pprint.PrettyPrinter( )

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.Qtable = {}
        self.Q_0 = 15
        self.Q_init = {None : self.Q_0, 'forward' : self.Q_0, 'left' : self.Q_0, 'right' : self.Q_0}
        self.gamma = 0.2 
        self.alpha =   0.9
        self.state = None
        self.prev_state  = None
        self.prev_action = None
        self.prev_reward = None
        
        if 1:
            self.load()
            print(len(self.Qtable))
            pp.pprint(self.Qtable)

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t, tr_no):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self) 
        # TODO: Update state
        self.state = []
        self.state.append(self.next_waypoint)
        self.state.append(inputs['light'])
        self.state.append(inputs['oncoming'])
        self.state.append(inputs['left']) 
        state = tuple(self.state)		
        # TODO: Select action according to your policy
        if self.Qtable.has_key(state) :  #check if state has been encountered before or not
            if random.random() < 1:  
                #pull the best action, or best actions if there are more than one with a max Q' value
                Q_max = max(self.Qtable[state].values())
                actions = {action:Q for action, Q in self.Qtable[state].items() if Q == Q_max}
                action = random.choice(actions.keys()) #pick one if there are multiple actions with same Q_max
            else : # if random float eclipses epsilon, choose a random action.
                action = random.choice([None, 'forward', 'left', 'right'])
        else :  #state has never been encountered
            self.Qtable[state] = self.Q_init.copy() #Add state to Qtable dictionary
            action = random.choice([None, 'forward', 'left', 'right'])  #choose one of the actions at random

        # Execute action and get reward
        reward = self.env.act(self, action)

        # TODO: Learn policy based on state, action, reward
        '''     Q = (1-alpha) Q_prev  + alpha* ( R + gamma max( Q_all_states)  )   '''
        if self.prev_state is not None :  #make sure it is not the first step in a trial.
			prev_state = tuple(self.prev_state)           
			Qp = (1 - self.alpha)*self.Qtable[prev_state][self.prev_action] \
                +  self.alpha*( self.prev_reward + self.gamma * max(self.Qtable[state].values()) )
			self.Qtable[prev_state][self.prev_action] = Qp

        #Store actions, state and reward as _prev for use in the next cycle
        self.prev_state  = self.state
        self.prev_action = action
        self.prev_reward = reward
        
        print "SIZE OF QTABLE  ", len(self.Qtable), " trial-no   ", tr_no+1
        #~ pp.pprint(self.Qtable)
        # print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        
        
    def save(self):
        name="QT.pickle"
        
        with open(name, 'wb') as handle:
            pickle.dump(self.Qtable, handle) 
    
    
    
    def load(self):
        name="QT.pickle"
       
        with open(name , 'rb') as handle:
            self.Qtable = pickle.load(handle)
        


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=False)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.00001, display=False )  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=11)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    
    #~ e.primary_agent.save()
    
    


if __name__ == '__main__':
    run()
