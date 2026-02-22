        # start learning
        #*The observations are being computed in unitree_rl_mjlab/mjlab/rl/vecenv_wrapper.py
        obs = self.env.get_observations().to(self.device)
        
        #TODO Would need to change what observations are being recieved here - INSPECT Observations first:
        #*The observation contains the input for the policy and the critic atm, the critic has 48 observations
        #*per agent, while the policy has 45, What are the three extra observations for the critic? 
        #TODO The crific contains 7 "terms" i.e. lin_vel, ang_vel, gravity etc. while the policy contains 6 terms, not the base_lin_vel
        #TODO hence critic contains 48 observations(45 + 3 values for the base_lin_vel)
        
        
        #? PROPRIOCETIVE OBSERVATIONS:
        # Body vel - lin + ang [X]
        #! Orientation - [X]
        # Joint position - [X]
        # Velocty HISTORY - [Can be done from buffer, but then what is sent initially? Empty tensor - mp]
        # ACTION HISTORY - [LAST ACTION AVAILABLE - GET HISTORY FROM BUFFER]
        #! LEG'S PHASE  - 
        
        #? EXTEROCEPTIVE OBSERVATIONS:
        #! Heightmap arond the robot - [X]
        
        #? PRIVIELLGED OBSERVATIONS(CHECK REQUIRED):
        #! CONTACT STATES [X]
        #! CONTACT FORCES [X]
        #! CONTACT NORMALS [X]
        #! FRICTION COEFFICIENTS - CHange needed
        #! THIGH AND SHANK CONTACT STATES [X]
        #! EXTERNAL FORCES AND TORQUES ON THE BODY [X]

        # Questions to ask when adding sensors - Are they available as builtin sensors?
        - If yes, fetch, if not - can two or more builtin sensors give us those values?
        - If not, can we add the new sensor?
        - If not - we drop it

        - Regarding histories - need to figure out the size of the history buffer