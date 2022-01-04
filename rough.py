import torch

def sort_categories(data, top_k):
    """  
    @params:
    - data: (d,n,num_categories)
    """
    sums = []
    num_categories = data.shape[-1]

    for i in range(num_categories):
        sums += [torch.sum(data[:,:,i])]

    sums = torch.tensor(sums)
    indices = torch.sort(sums, descending=True).indices

    return data[:,:,indices[:top_k]]

labels = torch.tensor(
        [
            # sentence 1
            [
                [1,0,0,1,0], # word 1
                [1,1,0,1,0], # word 2
                [0,0,0,1,0]  # word 3
            ],
            # sentence 2
            [
                [0,0,0,1,1],
                [1,0,1,1,0],
                [0,1,0,1,0]
            ]
        ] 
    )

print(sort_categories(labels, 2))
