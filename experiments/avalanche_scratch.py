from torch import mode
from exp_framework.learning import Net

from torch.optim import SGD, Adam
from torch.nn import CrossEntropyLoss
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics, forgetting_metrics
from avalanche.logging import InteractiveLogger, TensorboardLogger, TextLogger
from avalanche.training.supervised import Naive, CWRStar, Replay, GDumb, Cumulative, LwF, GEM, AGEM, EWC  # and many more!
from avalanche.benchmarks.classic import RotatedMNIST, SplitMNIST
from avalanche.training.plugins import CWRStarPlugin, ReplayPlugin, EWCPlugin, TrainGeneratorAfterExpPlugin, LwFPlugin
from avalanche.training.plugins import EvaluationPlugin
import pprint

model = Net(input_dim=28 * 28, output_dim=10)
# model = SimpleMLP(num_classes=10)
optimize = Adam(model.parameters(), lr=0.001)

evp = EvaluationPlugin(
    accuracy_metrics(minibatch=True),
    # forgetting_metrics(experience=True),
    loggers=[InteractiveLogger(), TextLogger(open('log.txt', 'a'))]
)

replay = ReplayPlugin(mem_size=100)
cwr = CWRStarPlugin(model=model)
ewc = EWCPlugin(ewc_lambda=0.001)
tga = TrainGeneratorAfterExpPlugin()
lwf = LwFPlugin()

cl_strategy = Naive(
    model, optimizer=optimize, criterion=CrossEntropyLoss(),
    train_mb_size=128, train_epochs=1, eval_mb_size=128,
    plugins=[evp]
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

    data_size = len(experience.dataset)
    x = experience.dataset
    first_data = x[0]
    print("Start of experience: ", experience.current_experience)
    print("Current Classes: ", experience.classes_in_this_experience)

    cl_strategy.train(experience)
    print('Training completed')

    print('Computing accuracy on the whole test set')
    results.append(cl_strategy.eval(benchmark.test_stream))

for r in results:
    pprint.pprint(r)

pprint.pprint(evp.get_all_metrics())
# print(results)