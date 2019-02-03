# distutils: language = c++
# cython: language_level=3, boundscheck=True

# todo Convert to cpp and wrap
# todo 

# todo Convert to delegate/composition pattern
import numpy as np
from cpython cimport bool
from cython.view cimport array as cvarray
from cython.operator cimport address
from libcpp.vector cimport vector
from libc cimport math as cmath
from  temp_array cimport TempArray
from libcpp.algorithm cimport partial_sort

cimport cython

# Define module level temporary array manager for fast creation of short-lived
# c++ arrays
cdef vector[double] def_buf = vector[double](0)
cdef vector[double]* buf = address(def_buf)

cdef extern from "math.h":
    double sqrt(double m)

cdef cython.numeric sqr(cython.numeric x):
    return x*x
    

cdef class RoverDomain:
    cdef public Py_ssize_t n_rovers
    cdef public Py_ssize_t n_pois
    cdef public Py_ssize_t n_steps
    cdef public Py_ssize_t n_req
    cdef public Py_ssize_t n_obs_sections
    cdef public double min_dist
    cdef public Py_ssize_t step_id
    cdef public bool done
    cdef public double setup_size
    cdef public double interaction_dist
    cdef public bool reorients
    cdef public bool discounts_eval
    cdef public double[:, :] init_rover_positions
    cdef public double[:, :] init_rover_orientations
    cdef public double[:, :]rover_positions
    cdef public double[:, :, :] rover_position_histories
    cdef public double[:, :] rover_orientations
    cdef public double[:] poi_values
    cdef public double[:, :] poi_positions
    
    # Some return values are stored for performance reasons
    cdef public double[:, :, :] rover_observations
    cdef public double[:] rover_rewards
    cdef public str reward_structure
    cdef public bool poi_turnoff
    cdef public double[:] poi_status
    cdef public double[:] rover_local_rewards
    cdef public bool poi_disappear
    
    
    def __cinit__(self):
        self.n_rovers = 1
        self.n_pois = 2
        self.n_steps = 15
        
        self.n_req = 1
        self.min_dist = 1.
        self.step_id = 0
        
        # Done is set to true to prevent stepping the sim forward 
        # before a call to reset (i.e. the domain is not yet ready/initialized).
        self.done = True 
        
        self.setup_size = 10.
        self.interaction_dist = 2.
        self.n_obs_sections = 36
        self.reorients = False
        self.discounts_eval = False
        
        # Set positions to zero lets rover domain generate self for you
        self.init_rover_positions = None
        self.init_rover_orientations =  None
        self.rover_positions =  None
        self.rover_position_histories =  None
        self.rover_orientations = None
            
        self.poi_values =  None
        self.poi_positions =  None
        self.rover_observations = None
        self.rover_rewards = None
        self.rover_local_rewards = np.zeros(self.n_rovers)

        self.reward_structure = 'soft_distance'
        self.poi_turnoff = True
        self.poi_status = np.ones(self.n_pois)
        self.poi_disappear = True


                        
    cpdef void reset(self):
        # Reset is the only function that allocates
        # permanent (but managed) memory

        self.poi_status = np.ones(self.n_pois)

        self.step_id = 0
        self.done = False
        
        # If user has not specified initial data, the domain provides
        # automatic initialization 
        if self.init_rover_positions is None:
            self.init_rover_positions = 0.5 * self.setup_size * np.ones(
                (self.n_rovers, 2))  
        if self.init_rover_orientations is None:
            rand_angles = np.random.uniform(-np.pi, np.pi, self.n_rovers)
            self.init_rover_orientations = np.vstack((np.cos(rand_angles),
                np.sin(rand_angles))).T
        if self.poi_values is None:
            self.poi_values = np.arange(self.n_pois) + 1.
        if self.poi_positions is None:
            self.poi_positions = (np.random.rand(self.n_pois, 2) * 
                self.setup_size)
    
        # If initial data is invalid (e.g. number of initial rovers does not  
        # match init_rover_positions.shape[0]), we need to raise an error
        if self.init_rover_positions.shape[0] != self.n_rovers:
            raise ValueError(
                'Number of rovers does not match number of initial ' 
                + 'rover positions')
        if self.init_rover_orientations.shape[0] != self.n_rovers:
            raise ValueError(
                'Number of rovers does not match number of initial ' 
                + 'rover orientations')
        if self.poi_values.shape[0] != self.n_pois:
            raise ValueError(
                'Number of POIs does not match number of POI values')
        if self.poi_positions.shape[0] != self.n_pois:
            raise ValueError(
                'Number of POIs does not match number of POI positions')
         
        # Create all unspecified working data
        if self.rover_positions is  None:
            self.rover_positions = np.zeros((self.n_rovers, 2))
        if self.rover_orientations is  None:
            self.rover_orientations = np.zeros((self.n_rovers, 2))
        if self.rover_position_histories is None:
            self.rover_position_histories = np.zeros((self.n_steps + 1, 
                self.n_rovers, 2))
        
        # Create all unspecified return data
        if self.rover_observations is None:
            self.rover_observations = np.zeros((self.n_rovers, 2, 
                self.n_obs_sections))
        if self.rover_rewards is None:
            self.rover_rewards = np.zeros(self.n_rovers)
            
        
        # Recreate all invalid working data
        if self.rover_positions.shape[0] != self.n_rovers:
            self.rover_positions = np.zeros((self.n_rovers, 2))
        if self.rover_orientations.shape[0] != self.n_rovers:
            self.rover_orientations = np.zeros((self.n_rovers, 2))
        if (self.rover_position_histories.shape[0] != self.n_steps + 1
                or self.rover_position_histories.shape[1] != self.n_rovers):
            self.rover_position_histories = np.zeros((self.n_steps + 1, 
                self.n_rovers, 2))
        
        # Recreate all invalid return data
        if (self.rover_observations.shape[0] != self.n_rovers or
                self.rover_observations.shape[3] != self.n_obs_sections):
            self.rover_observations = np.zeros((self.n_rovers, 2,
                self.n_obs_sections))
        if self.rover_rewards.shape[0] != self.n_rovers:
            self.rover_rewards = np.zeros(self.n_rovers)
        
        # Copy over initial data to working data
        self.rover_positions[...] = self.init_rover_positions
        self.rover_orientations[...] = self.init_rover_orientations
        
        # Store first rover positions in histories
        # todo avoiding slicing for speed?
        self.rover_position_histories[0,...] = self.init_rover_positions
            
    cpdef void stop_prematurely(self):
        self.n_steps = self.step_id
        self.done = True

    def step(self, double[:, :] actions, evaluate = None):
        """
        Provided for convenience, not recommended for performance
        """
        self.rover_local_rewards = np.zeros(self.n_rovers)
        if not self.done:
            if actions is not None:
                self.move_rovers(actions)
            self.step_id += 1
            
            # We must record rover positions after increasing the step 
            # index because the initial position before any movement
            # is stored in rover_position_histories[0], so the first step
            # (step_id = 0) must be stored in rover_position_histories[1]
            self.rover_position_histories[self.step_id,...]\
                = self.rover_positions
            
        self.done = self.step_id >= self.n_steps
        if evaluate:
            evaluate(self)
        else:
            for poi_id in range(self.n_pois):
                if self.poi_status[poi_id]:
                    self.update_local_step_reward_from_poi(poi_id)
        self.update_observations()

        return self.rover_observations, self.rover_local_rewards, self.done, self
    

    cpdef void move_rovers(self, double[:, :] actions):
        cdef Py_ssize_t rover_id
        cdef double dx, dy, norm
        
        # clip actions
        for rover_id in range(self.n_rovers):
            actions[rover_id, 0] = min(max(-1, actions[rover_id, 0]), 1)
            actions[rover_id, 1] = min(max(-1, actions[rover_id, 1]), 1)
        
        
        if self.reorients:
            # Translate and Reorient all rovers based on their actions
            for rover_id in range(self.n_rovers):
        
                # turn action into global frame motion
                dx = (self.rover_orientations[rover_id, 0]
                    * actions[rover_id, 0] 
                    - self.rover_orientations[rover_id, 1] 
                    * actions[rover_id, 1])
                dy = (self.rover_orientations[rover_id, 0] 
                    * actions[rover_id, 1] 
                    + self.rover_orientations[rover_id, 1] 
                    * actions[rover_id, 0])
                
            
                # globally move and reorient agent
                self.rover_positions[rover_id, 0] += dx
                self.rover_positions[rover_id, 1] += dy
                
                # Reorient agent in the direction of movement in the global 
                # frame.  Avoid divide by 0 (by skipping the reorientation step 
                # entirely).
                if not (dx == 0. and dy == 0.): 
                    norm = sqrt(dx*dx +  dy*dy)
                    self.rover_orientations[rover_id,0] = dx / norm
                    self.rover_orientations[rover_id,1] = dy / norm
        # Else domain translates but does not reorients agents
        else:
            for rover_id in range(self.n_rovers):
                self.rover_positions[rover_id, 0] += actions[rover_id, 0]
                self.rover_positions[rover_id, 1] += actions[rover_id, 1]
                







    cpdef double calc_step_eval_from_poi(self, Py_ssize_t poi_id):
        # todo profile the benefit (or loss) of TempArray
        cdef TempArray[double] sqr_dists_to_poi
        cdef double displ_x, displ_y, sqr_dist_sum
        cdef Py_ssize_t rover_id, near_rover_id
        
        sqr_dists_to_poi.alloc(buf, self.n_rovers)
        
        # Get the rover square distances to POIs.
        for rover_id in range(self.n_rovers):
            displ_x = (self.rover_positions[rover_id, 0]
                - self.poi_positions[poi_id, 0])
            displ_y = (self.rover_positions[rover_id, 1]
                - self.poi_positions[poi_id, 1])
            sqr_dists_to_poi[rover_id] = displ_x*displ_x + displ_y*displ_y
               
               
        # Sort (n_req) closest rovers for evaluation
        # Sqr_dists_to_poi is no longer in rover order!
        partial_sort(sqr_dists_to_poi.begin(), 
            sqr_dists_to_poi.begin() + self.n_req,
            sqr_dists_to_poi.end())


        
        # Is there (n_req) rovers observing? Only need to check the (n_req)th
        # closest rover
        if (sqr_dists_to_poi[self.n_req-1] > 
                self.interaction_dist * self.interaction_dist):
            # Not close enough?, then there is no reward for this POI
            return 0.

        #Reward POIs


        #Yes? Continue evaluation
        self.poi_status[poi_id] = False
        if self.discounts_eval:
            sqr_dist_sum = 0.
            # Get sum sqr distance of nearest rovers
            for near_rover_id in range(self.n_req):
                sqr_dist_sum += sqr_dists_to_poi[near_rover_id]

            if self.reward_structure == 'inv_distance':
                return self.poi_values[poi_id] / max(self.min_dist, sqr_dist_sum)
            elif self.reward_structure == 'soft-distance':
                return self.poi_values[poi_id] - 0.01 * sqr_dist_sum
        # Do not discount POI evaluation
        else:
            return self.poi_values[poi_id]




    cpdef double update_local_step_reward_from_poi(self, Py_ssize_t poi_id):
        # todo profile the benefit (or loss) of TempArray
        cdef TempArray[double] sqr_dists_to_poi
        cdef TempArray[double] srq_dists_to_poi_unsorted
        cdef double displ_x, displ_y, sqr_dist_sum, l_reward
        cdef Py_ssize_t rover_id, near_rover_id
        sqr_dists_to_poi.alloc(buf, self.n_rovers)
        srq_dists_to_poi_unsorted.alloc(buf, self.n_rovers)
        # Get the rover square distances to POIs.
        for rover_id in range(self.n_rovers):
            displ_x = (self.rover_positions[rover_id, 0]
                - self.poi_positions[poi_id, 0])
            displ_y = (self.rover_positions[rover_id, 1]
                - self.poi_positions[poi_id, 1])
            sqr_dists_to_poi[rover_id] = displ_x*displ_x + displ_y*displ_y
            srq_dists_to_poi_unsorted[rover_id] = sqr_dists_to_poi[rover_id]

        # Sort (n_req) closest rovers for evaluation
        # Sqr_dists_to_poi is no longer in rover order!
        partial_sort(sqr_dists_to_poi.begin(),
            sqr_dists_to_poi.begin() + self.n_req,
            sqr_dists_to_poi.end())

        # Is there (n_req) rovers observing? Only need to check the (n_req)th
        # closest rover
        if (sqr_dists_to_poi[self.n_req-1] >
                self.interaction_dist * self.interaction_dist):
            # Not close enough?, then there is no reward for this POI
            return 0.

        self.poi_status[poi_id] = False
        if self.discounts_eval:
            sqr_dist_sum = 0.
            # Get sum sqr distance of nearest rovers
            for near_rover_id in range(self.n_req):
                sqr_dist_sum += sqr_dists_to_poi[near_rover_id]
            l_reward = self.poi_values[poi_id] / max(self.min_dist,
                sqr_dist_sum)

        # Do not discount POI evaluation
        else:
            l_reward = self.poi_values[poi_id]


        #Yes? Continue evaluation
        for rover_id in range(self.n_rovers):
            for closest_rover_id in range(self.n_req):
                if sqr_dists_to_poi[closest_rover_id] == (srq_dists_to_poi_unsorted[rover_id]):
                    self.rover_local_rewards[rover_id] += l_reward
                    sqr_dists_to_poi[closest_rover_id] = -1.

        return 0.




    cpdef double calc_step_global_eval(self):
        cdef double eval
        cdef Py_ssize_t poi_id
        
        eval = 0.
        for poi_id in range(self.n_pois):
            if self.poi_status[poi_id]:
                eval += self.calc_step_eval_from_poi(poi_id)
        
        return eval
      
    cpdef double calc_step_cfact_global_eval(self, Py_ssize_t rover_id):
        # Hack: simulate counterfactual by moving agent FAR AWAY, then calculate
        
        cdef double actual_x, actual_y, far, eval
        far = 1000000. # That's far enough, right?
        
        # Store actual positions for later reassignment
        actual_x = self.rover_positions[rover_id, 0]
        actual_y = self.rover_positions[rover_id, 1]
        
        # Move rover artificially
        self.rover_positions[rover_id, 0] = far
        self.rover_positions[rover_id, 1] = far
        
        # Calculate /counterfactual/ evaluation
        eval = self.calc_step_global_eval()
        
        # Move rover back
        self.rover_positions[rover_id, 0] = actual_x
        self.rover_positions[rover_id, 1] = actual_y
        
        return eval
        

    cpdef double calc_traj_global_eval(self):
        cdef Py_ssize_t step_id, poi_id
        cdef TempArray[double] poi_evals
        cdef double eval
        
        # Only evaluate trajectories at the end
        if not self.done:
            return 0.
            
        # Initialize evaluations to 0
        eval = 0.
        poi_evals.alloc(buf, self.n_pois)
        for poi_id in range(self.n_pois):
            poi_evals[poi_id] = 0
        
        
        # Get evaluation for poi, for each step, storing the max
        for step_id in range(self.n_steps+1):
            # Go back in time
            self.rover_positions[...] = \
                self.rover_position_histories[step_id, ...]
            
            # Keep best step eval for each poi
            for poi_id in range(self.n_pois):
                poi_evals[poi_id] = max(poi_evals[poi_id],
                    self.calc_step_eval_from_poi(poi_id))
        
        
        # Set evaluation to the sum of all POI-specific evaluations
        for poi_id in range(self.n_pois):
            eval += poi_evals[poi_id]
        
        return eval
       

    cpdef double calc_traj_cfact_global_eval(self, Py_ssize_t rover_id):
        # Hack: simulate counterfactual by moving agent FAR AWAY, then calculate
        cdef TempArray[double] actual_x_hist, actual_y_hist,
        cdef double  far, eval
        cdef Py_ssize_t step_id
        far = 1000000. # That's far enough, right?
        
        actual_x_hist.alloc(buf, self.n_steps+1)
        actual_y_hist.alloc(buf, self.n_steps+1)
        
        for step_id in range(self.n_steps+1):
            # Store actual positions for later reassignment
            actual_x_hist[step_id] = \
                self.rover_position_histories[step_id, rover_id, 0]
            actual_y_hist[step_id] = \
                self.rover_position_histories[step_id, rover_id, 1]
            
            # Move rover artificially
            self.rover_position_histories[step_id, rover_id, 0] = far
            self.rover_position_histories[step_id, rover_id, 1] = far
        
        # Calculate /counterfactual/ evaluation
        eval = self.calc_traj_global_eval()
        
        for step_id in range(self.n_steps+1):
            # Move rover back
            self.rover_position_histories[step_id, rover_id, 0] = \
                actual_x_hist[step_id]
            self.rover_position_histories[step_id, rover_id, 1] = \
                actual_y_hist[step_id]

                
        return eval

    cpdef void add_to_sensor(self, Py_ssize_t rover_id, 
        Py_ssize_t type_id, double other_x, double other_y, double val):
            
            cdef double gf_displ_x, gf_displ_y, displ_x, displ_y, 
            cdef double rf_displ_x, rf_displ_y, dist, angle,  sec_id_temp
            cdef Py_ssize_t sec_id
            
            # Get global (gf) frame displacement
            gf_displ_x = (other_x - self.rover_positions[rover_id, 0])
            gf_displ_y = (other_y - self.rover_positions[rover_id, 1])
            
            # Set displacement value used by sensor to global frame
            displ_x = gf_displ_x
            displ_y = gf_displ_y
            
            # /May/ reorient displacement for observations
            if self.reorients:
                # Get rover frame (rf) displacement
                rf_displ_x = (self.rover_orientations[rover_id, 0] 
                    * displ_x
                    + self.rover_orientations[rover_id, 1]
                    * displ_y)
                rf_displ_y = (self.rover_orientations[rover_id, 0]
                    * displ_y
                    - self.rover_orientations[rover_id, 1]
                    * displ_x)
                # Set displacement value used by sensor to rover frame
                displ_x = rf_displ_x
                displ_y = rf_displ_y
                
            dist = cmath.sqrt(displ_x*displ_x + displ_y*displ_y)
                
            # By bounding distance value we 
            # implicitly bound sensor values
            if dist < self.min_dist:
                dist = self.min_dist
            
            # Get arc tangent (angle) of displacement 
            angle = cmath.atan2(displ_y, displ_x) 
            
            #  Get intermediate Section Index by discretizing angle
            sec_id_temp = cmath.floor(
                (angle + cmath.pi)
                / (2 * cmath.pi) 
                * self.n_obs_sections)
                
            # Clip and convert to get Section id
            sec_id = <Py_ssize_t>min(max(0, sec_id_temp), self.n_obs_sections-1)
                
            
            self.rover_observations[rover_id,type_id,sec_id] += val/(dist*dist)
        
    cpdef void update_observations(self):
        
        cdef Py_ssize_t rover_id, poi_id, other_rover_id
        
        # Zero all observations
        for rover_id in range(self.n_rovers):
            for type_id in range(2):
                for section_id in range(self.n_obs_sections):
                    self.rover_observations[rover_id, type_id, section_id] = 0.
        
        
        for rover_id in range(self.n_rovers):

            
            # Update rover type observations
            for other_rover_id in range(self.n_rovers):
                
                # agents do not sense self (ergo skip self comparison)
                if rover_id == other_rover_id:
                    continue

                self.add_to_sensor(rover_id, 0, 
                    self.rover_positions[rover_id, 0], 
                    self.rover_positions[rover_id, 1], 1.)    

            # Update POI type observations
            for poi_id in range(self.n_pois):

                if self.poi_disappear and self.poi_status[poi_id] == False: continue
            
                self.add_to_sensor(rover_id, 1,
                    self.poi_positions[poi_id, 0], 
                    self.poi_positions[poi_id, 1], 1.) 
                    
    
    
    cpdef void update_rewards_step_global_eval(self):
        cdef double global_eval
        cdef Py_ssize_t rover_id
        self.calc_step_global_eval()
    
    
    cpdef void update_rewards_step_diff_eval(self):
        cdef double global_eval, cfact_global_eval
        cdef Py_ssize_t rover_id
        
        global_eval = self.calc_step_global_eval()
        for rover_id in range(self.n_rovers):
            cfact_global_eval = self.calc_step_cfact_global_eval(rover_id)
            self.rover_rewards[rover_id] = global_eval - cfact_global_eval
    
    cpdef void update_rewards_traj_global_eval(self):
        cdef double global_eval
        cdef Py_ssize_t rover_id
        
        global_eval = self.calc_traj_global_eval()
        for rover_id in range(self.n_rovers):
            self.rover_rewards[rover_id] = global_eval
    
    cpdef void update_rewards_traj_diff_eval(self):
        cdef double global_eval, cfact_global_eval
        cdef Py_ssize_t rover_id
        
        global_eval = self.calc_traj_global_eval()
        for rover_id in range(self.n_rovers):
            cfact_global_eval = self.calc_traj_cfact_global_eval(rover_id)
            self.rover_rewards[rover_id] = global_eval - cfact_global_eval
        
        
        