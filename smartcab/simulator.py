import os
import time
import random
import importlib
import csv

import numpy as np

from analysis import Reporter

metric_TR_pt                      = "Total Reward per Trial"
metric_STD_pt                     = "Steps To Deadline per Trial"
metric_TNR_pt                     = "Total Negative Reward (TNR) per Trial"
metric_TNR_pspt                   = "TNR per Step per Trial"
metric_states                     = "Number of States Visited by the Simulation"
metric_CAVG                       = "Rolling Average (over all past Trials)"


class Simulator(object):
    """Simulates agents in a dynamic smartcab environment.

    Uses PyGame to display GUI, if available.
    """

    colors = {
        'black'   : (  0,   0,   0),
        'white'   : (255, 255, 255),
        'red'     : (255,   0,   0),
        'green'   : (  0, 255,   0),
        'dgreen'  : (  0, 228,   0),
        'blue'    : (  0,   0, 255),
        'cyan'    : (  0, 200, 200),
        'magenta' : (200,   0, 200),
        'yellow'  : (255, 255,   0),
        'mustard' : (200, 200,   0),
        'orange'  : (255, 128,   0),
        'maroon'  : (228,   0,   0),
        'gray'    : (155, 155, 155)
    }

    def __init__(self, env, size=None, update_delay=1.0, display=True,  live_plot=True):
        self.env = env
        self.size = size if size is not None else ((self.env.grid_size[0] + 1) * self.env.block_size, (self.env.grid_size[1] + 1) * self.env.block_size)
        self.width, self.height = self.size
        self.road_width = 44
        self.bg_color = self.colors['gray']
        self.road_color = self.colors['black']
        self.line_color = self.colors['mustard']
        self.stop_color = self.colors['red']
        self.go_color = self.colors['green']
        self.boundary = self.colors['black']

        self.quit = False
        self.start_time = None
        self.current_time = 0.0
        self.last_updated = 0.0
        self.update_delay = update_delay  # duration between each step (in secs)

        self.display = display
        if self.display:
            try:
                self.pygame = importlib.import_module('pygame')
                self.pygame.init()
                self.screen = self.pygame.display.set_mode(self.size)

                self.frame_delay = max(1, int(self.update_delay * 1000))  # delay between GUI frames in ms (min: 1)
                self.agent_sprite_size = (32, 32)
                self.agent_circle_radius = 20  # radius of circle, when using simple representation
                for agent in self.env.agent_states:
                    agent._sprite = self.pygame.transform.smoothscale(self.pygame.image.load(os.path.join("images", "car-{}.png".format(agent.color))), self.agent_sprite_size)
                    agent._sprite_size = (agent._sprite.get_width(), agent._sprite.get_height())

                self.font = self.pygame.font.Font(None, 28)
                self.paused = False
            except ImportError as e:
                self.display = False
                print "Simulator.__init__(): Unable to import pygame; display disabled.\n{}: {}".format(e.__class__.__name__, e)
            except Exception as e:
                self.display = False
                print "Simulator.__init__(): Error initializing GUI objects; display disabled.\n{}: {}".format(e.__class__.__name__, e)
        
        self.live_plot = live_plot
        
        self.rep0 = Reporter(metrics=[metric_TR_pt, metric_CAVG], live_plot=self.live_plot)
        self.rep1 = Reporter(metrics=[metric_STD_pt, metric_CAVG], live_plot=self.live_plot)
        self.rep2 = Reporter(metrics=[metric_TNR_pt,metric_CAVG], live_plot=self.live_plot) 
        self.rep3 = Reporter(metrics=[metric_TNR_pt ], live_plot=self.live_plot) 
        self.rep4 = Reporter(metrics=[metric_TNR_pspt], live_plot=self.live_plot)  
        self.rep5 = Reporter(metrics=[metric_states], live_plot=self.live_plot) 
        
        self.avg_net_reward_window = 0

    def run(self, n_trials=1):
        self.quit = False
        
        self.rep0.reset()
        self.rep1.reset() 
        self.rep2.reset() 
        self.rep3.reset() 
        self.rep4.reset() 
        self.rep5.reset() 
        
        for trial in xrange(n_trials):
            print "Simulator.run(): Trial {}".format(trial)  # [debug]
            self.env.reset()
            self.current_time = 0.0
            self.last_updated = 0.0
            self.start_time = time.time()
            while True:
                try:
                    # Update current time
                    self.current_time = time.time() - self.start_time
                    #print "Simulator.run(): current_time = {:.3f}".format(self.current_time)

                    # Handle GUI events
                    if self.display:
                        for event in self.pygame.event.get():
                            if event.type == self.pygame.QUIT:
                                self.quit = True
                            elif event.type == self.pygame.KEYDOWN:
                                if event.key == 27:  # Esc
                                    self.quit = True
                                elif event.unicode == u' ':
                                    self.paused = True

                        if self.paused:
                            self.pause()

                    # Update environment
                    if self.current_time - self.last_updated >= self.update_delay:
                        self.env.step(trial)
                        self.last_updated = self.current_time

                    # Render GUI and sleep
                    if self.display:
                        self.render()
                        self.pygame.time.wait(self.frame_delay)
                except KeyboardInterrupt:
                    self.quit = True
                finally:
                    if self.quit or self.env.done:
                        break

            if self.quit:
                break


            # Collect/update metrics
            self.rep0.collect(metric_TR_pt, trial, self.env.trial_data['net_reward'])  # total reward obtained in this trial
            self.rep0.collect(metric_CAVG, trial, np.mean( self.rep0.metrics[metric_TR_pt].ydata[:]))  # rolling mean of reward
           
            self.rep1.collect(metric_STD_pt, trial, self.env.trial_data['final_deadline'])  # final deadline value (time remaining)
            self.rep1.collect(metric_CAVG, trial, np.mean( self.rep1.metrics[metric_STD_pt].ydata ) )  # final deadline value (time remaining)

	    self.rep2.collect(metric_TNR_pt, trial, self.env.trial_data['negative_reward'])  # final deadline value (time remaining)
            self.rep2.collect(metric_CAVG, trial, np.mean( self.rep2.metrics[metric_TNR_pt].ydata ) )  # final deadline value (time remaining)
 
            self.rep3.collect(metric_TNR_pt, trial,  self.env.trial_data['negative_reward']  )     # final deadline value (time remaining)
            self.rep4.collect(metric_TNR_pspt, trial,  self.env.trial_data['negative_reward']/self.env.trial_data['final_deadline']  )     # final deadline value (time remaining)

            self.rep5.collect(metric_states, trial, len(self.env.primary_agent.Qtable) )
            
            if self.live_plot:
                self.rep0.refresh_plot()  # autoscales axes, draws stuff and flushes events
                self.rep1.refresh_plot()  # autoscales axes, draws stuff and flushes events
                self.rep2.refresh_plot()  # autoscales axes, draws stuff and flushes events
                self.rep3.refresh_plot()  # autoscales axes, draws stuff and flushes events
                self.rep4.refresh_plot()  # autoscales axes, draws stuff and flushes events
                self.rep5.refresh_plot()  # autoscales axes, draws stuff and flushes events

        # Report final metrics
        if self.display:
            self.pygame.display.quit()  # need to shutdown pygame before showing metrics plot
            # TODO: Figure out why having both game and plot displays makes things crash!

        if self.live_plot:
            self.rep0.show_plot()  # holds till user closes plot window
            self.rep1.show_plot()  # holds till user closes plot window
            self.rep2.show_plot()  # holds till user closes plot window
            self.rep3.show_plot()  # holds till user closes plot window
            self.rep4.show_plot()  # holds till user closes plot window
            self.rep5.show_plot()  # holds till user closes plot window
            

    def render(self):
        # Clear screen
        self.screen.fill(self.bg_color)

        # Draw elements
        # * Static elements
        for road in self.env.roads:
            self.pygame.draw.line(self.screen, self.road_color, (road[0][0] * self.env.block_size, road[0][1] * self.env.block_size), (road[1][0] * self.env.block_size, road[1][1] * self.env.block_size), self.road_width)

        for intersection, traffic_light in self.env.intersections.iteritems():
            self.pygame.draw.circle(self.screen, self.road_color, (intersection[0] * self.env.block_size, intersection[1] * self.env.block_size), 10)
            if traffic_light.state:  # North-South is open
                self.pygame.draw.line(self.screen, self.colors['green'],
                    (intersection[0] * self.env.block_size, intersection[1] * self.env.block_size - 15),
                    (intersection[0] * self.env.block_size, intersection[1] * self.env.block_size + 15), self.road_width)
            else:  # East-West is open
                self.pygame.draw.line(self.screen, self.colors['green'],
                    (intersection[0] * self.env.block_size - 15, intersection[1] * self.env.block_size),
                    (intersection[0] * self.env.block_size + 15, intersection[1] * self.env.block_size), self.road_width)

        # * Dynamic elements
        for agent, state in self.env.agent_states.iteritems():
            # Compute precise agent location here (back from the intersection some)
            agent_offset = (2 * state['heading'][0] * self.agent_circle_radius, 2 * state['heading'][1] * self.agent_circle_radius)
            agent_pos = (state['location'][0] * self.env.block_size - agent_offset[0], state['location'][1] * self.env.block_size - agent_offset[1])
            agent_color = self.colors[agent.color]
            if hasattr(agent, '_sprite') and agent._sprite is not None:
                # Draw agent sprite (image), properly rotated
                rotated_sprite = agent._sprite if state['heading'] == (1, 0) else self.pygame.transform.rotate(agent._sprite, 180 if state['heading'][0] == -1 else state['heading'][1] * -90)
                self.screen.blit(rotated_sprite,
                    self.pygame.rect.Rect(agent_pos[0] - agent._sprite_size[0] / 2, agent_pos[1] - agent._sprite_size[1] / 2,
                        agent._sprite_size[0], agent._sprite_size[1]))
            else:
                # Draw simple agent (circle with a short line segment poking out to indicate heading)
                self.pygame.draw.circle(self.screen, agent_color, agent_pos, self.agent_circle_radius)
                self.pygame.draw.line(self.screen, agent_color, agent_pos, state['location'], self.road_width)
            if agent.get_next_waypoint() is not None:
                self.screen.blit(self.font.render(agent.get_next_waypoint(), True, agent_color, self.bg_color), (agent_pos[0] + 10, agent_pos[1] + 10))
            if state['destination'] is not None:
                self.pygame.draw.circle(self.screen, agent_color, (state['destination'][0] * self.env.block_size, state['destination'][1] * self.env.block_size), 6)
                self.pygame.draw.circle(self.screen, agent_color, (state['destination'][0] * self.env.block_size, state['destination'][1] * self.env.block_size), 15, 2)

        # * Overlays
        text_y = 10
        for text in self.env.status_text.split('\n'):
            self.screen.blit(self.font.render(text, True, self.colors['red'], self.bg_color), (100, text_y))
            text_y += 20

        # Flip buffers
        self.pygame.display.flip()

    def pause(self):
        abs_pause_time = time.time()
        pause_text = "[PAUSED] Press any key to continue..."
        self.screen.blit(self.font.render(pause_text, True, self.colors['cyan'], self.bg_color), (100, self.height - 40))
        self.pygame.display.flip()
        print pause_text  # [debug]
        while self.paused:
            for event in self.pygame.event.get():
                if event.type == self.pygame.KEYDOWN:
                    self.paused = False
            self.pygame.time.wait(self.frame_delay)
        self.screen.blit(self.font.render(pause_text, True, self.bg_color, self.bg_color), (100, self.height - 40))
        self.start_time += (time.time() - abs_pause_time)
