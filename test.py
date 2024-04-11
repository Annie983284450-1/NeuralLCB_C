import numpy as np
# when self.num_contexts (user input) <= self.num_samples (num of contexts in the whole dataset)

# len(test_indices) should be in [0,self.num_samples )
test_indices = np.array([10, 20, 10, 30, 40, 20, 50])
num_test_contexts = 3
i = np.unique(test_indices,return_index=True)[1]
# sorting might be unnecessary, cuz np.unique() would already return the indices 
print(f"i:{i}")
i.sort()
print(f"i:{i}")
test_ind = test_indices[i[:num_test_contexts]]
print(f'test_ind:{test_ind}')