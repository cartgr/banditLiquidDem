from exp_framework.learning import Net
from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!
from avalanche.benchmarks.classic import RotatedMNIST, SplitMNIST
import pprint

model = Net(input_dim=28 * 28, output_dim=10)
optimize = Adam(model.parameters(), lr=0.001)
cl_strategy = Naive(
    model, optimizer=optimize, criterion=CrossEntropyLoss(),
    train_mb_size=128, train_epochs=1, eval_mb_size=128
)
# optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
# model = SimpleMLP(num_classes=10)
# criterion = CrossEntropyLoss()
# cl_strategy = Naive(
#     model, SGD(model.parameters(), lr=0.001, momentum=0.9), criterion,
#     train_mb_size=100, train_epochs=4, eval_mb_size=100
# )


# scenario
# benchmark = RotatedMNIST(n_experiences=5, seed=1)
benchmark = SplitMNIST(n_experiences=5, fixed_class_order=list(range(10)), seed=1)

# TRAINING LOOP
print('Starting experiment...')
results = []
for experience in benchmark.train_stream:
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(benchmark.test_stream))

for r in results:
    pprint.pprint(r)
# print(results)