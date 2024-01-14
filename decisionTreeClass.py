from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import pydotplus
from IPython.display import Image
from six import StringIO


class decisionTree(DecisionTreeClassifier):
    def __init__(self, max_depth) -> None:
        super().__init__(max_depth=max_depth)
        self.learned_classifier = None
        self.prediction = None
        self.train_features = None

    def learn_tree(self, train_features, train_target):
        print(f"Training decision tree with a maximum depth of {self.max_depth}...")
        self.train_features = train_features
        self.learned_classifier = self.fit(train_features, train_target)
        print("Finished training!")
    
    def predict_target(self, test_features):
        print("Running prediction...")
        self.prediction = self.learned_classifier.predict(test_features)
        print(f"Prediction made!")

    def get_tree_accuracy(self, true_target):
        self.test_acc = metrics.accuracy_score(y_true=true_target, y_pred=self.prediction)
        print(f"Accuracy of the learned decision tree is: {self.test_acc}")

    def visualise_tree(self):
        dot_data = StringIO()

        export_graphviz(self.learned_classifier,
                out_file=dot_data,
                filled=True,
                rounded=True,
                special_characters=True,
                feature_names=self.train_features.columns,
                class_names=['0','1'])
        
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        Image(graph.create_png())
