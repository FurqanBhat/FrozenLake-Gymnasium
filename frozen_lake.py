import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
def run(episodes, is_training=True, render=False):
    env=gym.make('FrozenLake-v1', map_name="8x8", is_slippery=True, render_mode="human" if render else None)

    if is_training:
        q=np.zeros((env.observation_space.n, env.action_space.n)) #64x4
    else:
        f=open('frozen_lake.pk1','rb')
        q=pickle.load(f)
        f.close()
    

    learning_rate_a=0.9 #alpha
    discount_factor_g=0.9 #gamma
    epsilon=1 #1=100%randomness
    epsilon_decay_rate=0.0001
    rng=np.random.default_rng() #random number generator
    rewards_per_episode=np.zeros(episodes)

    for i in range(episodes):
        state=env.reset()[0]
        terminated=False
        truncated=False

        

        while(not terminated and not truncated):
            
            if rng.random()<epsilon and is_training:
                action=env.action_space.sample()
            else:
                action=np.argmax(q[state, :])

            
            new_state, reward, terminated, truncated, _ = env.step(action)

            #Temporal difference learning
            #sample=R(s,a,s')+gamma(V(s')
            #V(s)=(1-alpha)*V(s)+ alpha*(sample)
            #OR V(s)=V(s)+alpha*(sample-V(s))
        
            if is_training:
                q[state, action]=q[state, action]+learning_rate_a*(reward+discount_factor_g*np.max(q[new_state, :]) -q[state, action])

            state=new_state
            


        epsilon=max(epsilon-epsilon_decay_rate,0)
        if epsilon==0:
            learning_rate_a=0.0001  
        if reward==1:
            rewards_per_episode[i]=1   

        


    env.close()

    sum_rewards=np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t]=np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    plt.plot(sum_rewards)
    plt.xlabel("Episodes")
    plt.ylabel("Average Reward (Last 100 Episodes)")
    plt.savefig('frozen_lake8x8.png')

    if is_training:
        f=open('frozen_lake.pk1','wb')
        pickle.dump(q,f)
        f.close()
    

if __name__=='__main__':
    run(1500, is_training=False, render=False)