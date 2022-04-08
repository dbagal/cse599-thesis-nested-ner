content = """IL      B-DNA   O       O       O
-       I-DNA   O       O       O
2       I-DNA   O       O       O
gene    I-DNA   O       O       O
expression      O       O       O       O
and     O       O       O       O
NF      B-protein       O       O       O
-       I-protein       O       O       O
kappa   I-protein       O       O       O
B       I-protein       O       O       O
activation      O       O       O       O
through O       O       O       O
CD28    B-protein       O       O       O
requires        O       O       O       O
reactive        O       O       O       O
oxygen  O       O       O       O
production      O       O       O       O
by      O       O       O       O
5       B-protein       O       O       O
-       I-protein       O       O       O
lipoxygenase    I-protein       O       O       O
.       O       O       O       O"""

content = [line.split("\t") for line in content.split("\n")]
print(content)