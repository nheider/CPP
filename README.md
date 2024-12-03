REASON-MC: RL for Efficient Sampling Of spatial FunctioNs with Movement Constraints

High-level planner for robots to efficently approximate a spatial function from exposure to previous samples of the function, with the robots movement constaints. 

Status: Work in Progress. Will be significantly updated in the next couple of weeks
- Overall environment works, with hard-coded movement constraints and uniform distribution (i.e. Coverage Path Planning)
- Structure: Env contains env logic, train.py contains PPO stuff
- To do:
  implement a yaml, to specify movement constraints; implement a way to specify spatial functions; better evalution methods; experiments beyond PPO;multiple parallel environments; documentation 

Image: Agent needs a lot of training to even learn how to stay in bounds. 
![alt text](https://github.com/nheider/REASON-MC/blob/main/field_map_with_lidar.png?raw=true)
