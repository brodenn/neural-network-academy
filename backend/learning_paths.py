"""
Learning Paths for Neural Network Academy

Defines structured curricula guiding learners through the 28 problems
with clear progression, prerequisites, and educational scaffolding.
"""

# Wow, riktigt coolt!

from dataclasses import dataclass
from typing import List, Dict, Optional, Literal

# Step types for different interactive challenges
StepType = Literal['training', 'build_challenge', 'prediction_quiz', 'debug_challenge', 'tuning_challenge']

@dataclass
class BuildChallengeData:
    """Data for build_challenge step type"""
    min_layers: int = 0
    max_layers: int = 5
    min_hidden_neurons: int = 1
    max_hidden_neurons: int = 16
    must_have_hidden: bool = False
    success_message: str = "Great architecture!"

@dataclass
class PredictionQuizData:
    """Data for prediction_quiz step type"""
    quiz_id: str  # References PREDICTION_QUIZZES in frontend
    show_result_after: bool = True  # Show training result after quiz

@dataclass
class DebugChallengeData:
    """Data for debug_challenge step type"""
    challenge_id: str  # References DEBUG_CHALLENGES in frontend

@dataclass
class TuningChallengeData:
    """Data for tuning_challenge step type - find optimal hyperparameters"""
    parameter: str  # 'learning_rate', 'epochs', 'architecture'
    target_loss: float = 0.1
    max_attempts: int = 5

@dataclass
class PathStep:
    step_number: int
    problem_id: str
    title: str
    learning_objectives: List[str]
    required_accuracy: float = 0.95
    hints: List[str] = None
    # New: step type for interactive challenges
    step_type: StepType = 'training'
    # Challenge-specific data
    build_challenge: Optional[BuildChallengeData] = None
    prediction_quiz: Optional[PredictionQuizData] = None
    debug_challenge: Optional[DebugChallengeData] = None
    tuning_challenge: Optional[TuningChallengeData] = None

    def __post_init__(self):
        if self.hints is None:
            self.hints = []

@dataclass
class LearningPath:
    id: str
    name: str
    description: str
    difficulty: str  # 'beginner', 'intermediate', 'advanced', 'research'
    estimated_time: str
    badge_icon: str
    badge_title: str
    badge_color: str
    prerequisites: List[str]  # List of path IDs that must be completed first
    steps: List[PathStep]

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'difficulty': self.difficulty,
            'estimatedTime': self.estimated_time,
            'prerequisites': self.prerequisites,
            'steps': len(self.steps),
            'badge': {
                'icon': self.badge_icon,
                'title': self.badge_title,
                'color': self.badge_color
            }
        }

    def get_step_details(self) -> List[Dict]:
        """Get detailed information about all steps"""
        result = []
        for step in self.steps:
            step_dict = {
                'stepNumber': step.step_number,
                'problemId': step.problem_id,
                'title': step.title,
                'learningObjectives': step.learning_objectives,
                'requiredAccuracy': step.required_accuracy,
                'hints': step.hints,
                'stepType': step.step_type,
            }
            # Add challenge-specific data if present
            if step.build_challenge:
                step_dict['buildChallenge'] = {
                    'minLayers': step.build_challenge.min_layers,
                    'maxLayers': step.build_challenge.max_layers,
                    'minHiddenNeurons': step.build_challenge.min_hidden_neurons,
                    'maxHiddenNeurons': step.build_challenge.max_hidden_neurons,
                    'mustHaveHidden': step.build_challenge.must_have_hidden,
                    'successMessage': step.build_challenge.success_message,
                }
            if step.prediction_quiz:
                step_dict['predictionQuiz'] = {
                    'quizId': step.prediction_quiz.quiz_id,
                    'showResultAfter': step.prediction_quiz.show_result_after,
                }
            if step.debug_challenge:
                step_dict['debugChallenge'] = {
                    'challengeId': step.debug_challenge.challenge_id,
                }
            if step.tuning_challenge:
                step_dict['tuningChallenge'] = {
                    'parameter': step.tuning_challenge.parameter,
                    'targetLoss': step.tuning_challenge.target_loss,
                    'maxAttempts': step.tuning_challenge.max_attempts,
                }
            result.append(step_dict)
        return result


# Define all learning paths

LEARNING_PATHS = {
    'foundations': LearningPath(
        id='foundations',
        name='Foundations',
        description='Master basic neural network concepts from logic gates to hidden layers',
        difficulty='beginner',
        estimated_time='2-3 hours',
        badge_icon='🏆',
        badge_title='Foundation Scholar',
        badge_color='#3B82F6',
        prerequisites=[],
        steps=[
            PathStep(
                step_number=1,
                problem_id='and_gate',
                title='Does AND Need a Hidden Layer?',
                learning_objectives=[
                    'Understand what a single neuron can learn',
                    'See linear separability in action',
                    'Learn about activation functions'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='and_simple',
                    show_result_after=True
                ),
                hints=[
                    'A single neuron can learn linearly separable patterns',
                    'Try training with just a few epochs to see quick convergence'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='or_gate',
                title='How Does OR Differ from AND?',
                learning_objectives=[
                    'Recognize another linearly separable problem',
                    'Compare with AND gate behavior',
                    'Understand decision boundaries'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='or_vs_and',
                    show_result_after=True
                ),
                hints=[
                    'OR is also linearly separable like AND',
                    'Notice how similar the training is to AND'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='not_gate',
                title='Build the Simplest Network',
                learning_objectives=[
                    'See how neurons invert signals',
                    'Understand bias importance',
                    'Learn about single-input neurons'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=0,
                    max_layers=1,
                    min_hidden_neurons=0,
                    max_hidden_neurons=4,
                    must_have_hidden=False,
                    success_message='Perfect! NOT only needs a single neuron!'
                ),
                hints=[
                    'This is the simplest possible neural network',
                    'The bias term is crucial for this transformation'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='nand_gate',
                title='Why is NAND Universal?',
                learning_objectives=[
                    'Learn about NAND as a universal gate',
                    'See another linearly separable problem',
                    'Understand that NAND can build ANY logic circuit'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='nand_universal',
                    show_result_after=True
                ),
                hints=[
                    'NAND is called "universal" - you can build ANY logic from it!',
                    'Still solvable with a single neuron like AND/OR'
                ]
            ),
            PathStep(
                step_number=5,
                problem_id='xor',
                title='Can XOR Learn Without Hidden Layers?',
                learning_objectives=[
                    'Understand that XOR is NOT linearly separable',
                    'Learn why hidden layers are necessary',
                    'See how architecture affects what networks can learn'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='xor_no_hidden',
                    show_result_after=True
                ),
                hints=[
                    'XOR cannot be solved with a single neuron!',
                    'The default [2, 4, 1] architecture adds hidden layers',
                    'Hidden layers allow the network to learn non-linear patterns'
                ]
            ),
            PathStep(
                step_number=6,
                problem_id='xor',
                title='Build a Working XOR Network',
                learning_objectives=[
                    'Apply the lesson from the prediction quiz',
                    'Understand why XOR needs hidden layers',
                    'Build a working XOR network'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=1,
                    max_layers=3,
                    min_hidden_neurons=2,
                    max_hidden_neurons=8,
                    must_have_hidden=True,
                    success_message='Excellent! XOR requires hidden layers to work!'
                ),
                hints=[
                    'Remember: XOR is NOT linearly separable',
                    'You need at least one hidden layer',
                    'Try 2-4 neurons in the hidden layer'
                ]
            ),
            PathStep(
                step_number=7,
                problem_id='xnor',
                title='Build XNOR Architecture',
                learning_objectives=[
                    'Apply hidden layer knowledge to XNOR',
                    'Recognize XNOR is similar to XOR',
                    'Build confidence with non-linear problems'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=1,
                    max_layers=3,
                    min_hidden_neurons=2,
                    max_hidden_neurons=8,
                    must_have_hidden=True,
                    success_message='Excellent! XNOR needs hidden layers just like XOR!'
                ),
                hints=[
                    'XNOR is the opposite of XOR (1 when inputs match)',
                    'Same architecture as XOR should work well'
                ]
            ),
            PathStep(
                step_number=8,
                problem_id='xor',
                title='Why Won\'t It Learn?',
                learning_objectives=[
                    'Diagnose common neural network bugs',
                    'Understand the symmetry problem',
                    'Learn about weight initialization'
                ],
                step_type='debug_challenge',
                debug_challenge=DebugChallengeData(
                    challenge_id='zero_init_bug'
                ),
                hints=[
                    'Look at the weight initialization setting',
                    'What happens when all weights start the same?'
                ]
            ),
            PathStep(
                step_number=9,
                problem_id='xor_5bit',
                title='Right-Size for 5-Bit Parity',
                learning_objectives=[
                    'Apply XOR concept to 5 inputs instead of 2',
                    'See how networks scale to more complex problems',
                    'Learn about the parity problem'
                ],
                step_type='tuning_challenge',
                tuning_challenge=TuningChallengeData(
                    parameter='parity_capacity',
                    target_loss=0.1,
                    max_attempts=5
                ),
                hints=[
                    'Parity checks if the sum of bits is odd or even',
                    'This is XOR extended to 5 inputs!',
                    'Deeper networks help with this complexity'
                ]
            )
        ]
    ),

    'training-mastery': LearningPath(
        id='training-mastery',
        name='Training Mastery',
        description='Master training, initialization, and hyperparameters',
        difficulty='intermediate',
        estimated_time='3-4 hours',
        badge_icon='🧠',
        badge_title='Neural Navigator',
        badge_color='#8B5CF6',
        prerequisites=['foundations'],
        steps=[
            PathStep(
                step_number=1,
                problem_id='fail_zero_init',
                title='Predict: Zero Weight Initialization',
                learning_objectives=[
                    'Understand the symmetry breaking problem',
                    'Learn why initialization matters',
                    'See what happens without random weights'
                ],
                required_accuracy=0.0,
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='zero_init',
                    show_result_after=True
                ),
                hints=[
                    'All neurons learn the same thing with zero init!',
                    'Random initialization breaks symmetry',
                    'This is a fundamental requirement for learning'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='fail_lr_high',
                title='Predict: Learning Rate = 10',
                learning_objectives=[
                    'Understand learning rate effects',
                    'Predict gradient descent behavior',
                    'Learn about stability vs speed tradeoff'
                ],
                required_accuracy=0.0,
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='high_lr',
                    show_result_after=True
                ),
                hints=[
                    'Learning rate controls step size in gradient descent',
                    'What happens if steps are TOO big?'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='fail_lr_high',
                title='Debug: Loss Explodes!',
                learning_objectives=[
                    'Recognize learning rate problems',
                    'Diagnose numerical instability',
                    'Apply knowledge from prediction quiz'
                ],
                required_accuracy=0.0,
                step_type='debug_challenge',
                debug_challenge=DebugChallengeData(
                    challenge_id='exploding_loss'
                ),
                hints=[
                    'The symptoms tell you a lot',
                    'This relates to what you just predicted!'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='fail_lr_low',
                title='Predict: Learning Rate = 0.0001',
                learning_objectives=[
                    'Understand optimization speed',
                    'Learn about efficiency tradeoffs',
                    'See practical training issues'
                ],
                required_accuracy=0.0,
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='slow_vs_fast_lr',
                    show_result_after=True
                ),
                hints=[
                    'Very small learning rate means very small steps',
                    'The network IS learning, just extremely slowly'
                ]
            ),
            PathStep(
                step_number=5,
                problem_id='fail_lr_low',
                title='Debug: Training Barely Moves',
                learning_objectives=[
                    'Understand slow convergence',
                    'Learn about optimization speed',
                    'See the tradeoff between stability and speed'
                ],
                required_accuracy=0.0,
                step_type='debug_challenge',
                debug_challenge=DebugChallengeData(
                    challenge_id='glacial_learning'
                ),
                hints=[
                    'Training will be extremely slow',
                    'The network barely learns each epoch',
                    'Too low = taking tiny steps toward minimum'
                ]
            ),
            PathStep(
                step_number=6,
                problem_id='linear',
                title='Can Networks Learn a Straight Line?',
                learning_objectives=[
                    'Introduction to regression problems',
                    'Learn about continuous vs discrete outputs',
                    'Understand linear function approximation'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='regression_intro',
                    show_result_after=True
                ),
                hints=[
                    'The network outputs a continuous value, not a class',
                    'This is the simplest regression problem',
                    'Linear relationships are easiest to learn'
                ]
            ),
            PathStep(
                step_number=7,
                problem_id='sine_wave',
                title='Debug: Outputs Stuck Between 0-1',
                learning_objectives=[
                    'Learn about regression vs classification',
                    'Understand continuous outputs',
                    'See how activation functions affect output range'
                ],
                step_type='debug_challenge',
                debug_challenge=DebugChallengeData(
                    challenge_id='wrong_activation'
                ),
                hints=[
                    'The network learns to approximate a sine wave',
                    'Check the activation function',
                    'Sigmoid squashes outputs to (0, 1)'
                ]
            ),
            PathStep(
                step_number=8,
                problem_id='polynomial',
                title='Design a Regression Network',
                learning_objectives=[
                    'Master non-linear function approximation',
                    'Understand different regression complexities',
                    'Learn about universal approximation'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=1,
                    max_layers=3,
                    min_hidden_neurons=4,
                    max_hidden_neurons=16,
                    must_have_hidden=True,
                    success_message='Good architecture! Let\'s see how it approximates the polynomial!'
                ),
                hints=[
                    'Neural networks can approximate any continuous function',
                    'Polynomials are good test cases',
                    'Watch how well the network fits the curve'
                ]
            ),
            PathStep(
                step_number=9,
                problem_id='xor',
                title='Find the Minimum Epochs for XOR',
                learning_objectives=[
                    'Apply everything you learned',
                    'Find the optimal training budget for XOR',
                    'Understand training efficiency'
                ],
                step_type='tuning_challenge',
                tuning_challenge=TuningChallengeData(
                    parameter='xor_epochs',
                    target_loss=0.05,
                    max_attempts=5
                ),
                hints=[
                    'Use what you learned: hidden layers, good init, reasonable LR',
                    'How many epochs does XOR really need?'
                ]
            )
        ]
    ),

    'boundaries-and-classes': LearningPath(
        id='boundaries-and-classes',
        name='Boundaries & Classes',
        description='Master 2D decision boundaries and multi-class classification',
        difficulty='intermediate',
        estimated_time='3-4 hours',
        badge_icon='🎨',
        badge_title='Classifier Champion',
        badge_color='#8B5CF6',
        prerequisites=['foundations'],
        steps=[
            PathStep(
                step_number=1,
                problem_id='two_blobs',
                title='What Boundary Separates Two Clusters?',
                learning_objectives=[
                    'Introduction to 2D decision boundaries',
                    'Visualize how networks separate data in 2D',
                    'Build intuition before complex shapes'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='blob_boundary',
                    show_result_after=True
                ),
                hints=[
                    'Two clusters that are easy to separate',
                    'Watch the decision boundary form',
                    'This is a warm-up for harder 2D problems'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='two_blobs',
                title='Build a Blob Classifier',
                learning_objectives=[
                    'Design a network for 2D classification',
                    'Understand that simple problems need simple networks',
                    'Practice building network architectures'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=0,
                    max_layers=2,
                    min_hidden_neurons=2,
                    max_hidden_neurons=8,
                    must_have_hidden=False,
                    success_message='Good design! Simple blobs don\'t need complex networks.'
                ),
                hints=[
                    'Well-separated clusters can be classified simply',
                    'Do you even need a hidden layer for this?'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='moons',
                title='What Boundary Do Moons Need?',
                learning_objectives=[
                    'Handle complex non-linear boundaries',
                    'Understand curved decision surfaces',
                    'See advanced boundary shapes'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='boundary_shape',
                    show_result_after=True
                ),
                hints=[
                    'The two "moons" interlock in a complex way',
                    'The decision boundary must curve around them',
                    'This tests the network\'s flexibility'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='circle',
                title='Design for Circle Decision Boundary',
                learning_objectives=[
                    'Learn about radial decision boundaries',
                    'Design networks for non-convex shapes',
                    'Understand why hidden layers matter for circles'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=1,
                    max_layers=3,
                    min_hidden_neurons=4,
                    max_hidden_neurons=12,
                    must_have_hidden=True,
                    success_message='Great! Circles need hidden layers for the radial boundary!'
                ),
                hints=[
                    'One class is inside a circle, one outside',
                    'A circle is not linearly separable',
                    'Hidden layers help learn curved boundaries'
                ]
            ),
            PathStep(
                step_number=5,
                problem_id='circle',
                title='Find the Learning Rate Sweet Spot',
                learning_objectives=[
                    'Master hyperparameter tuning',
                    'Understand the LR sweet spot',
                    'Balance training speed and stability'
                ],
                step_type='tuning_challenge',
                tuning_challenge=TuningChallengeData(
                    parameter='lr_sweet_spot',
                    target_loss=0.05,
                    max_attempts=5
                ),
                hints=[
                    'One class is inside a circle, one outside',
                    'The learning rate controls training stability and speed',
                    'Try values between 0.1 and 1.0'
                ]
            ),
            PathStep(
                step_number=6,
                problem_id='quadrants',
                title='How Many Outputs for 4 Classes?',
                learning_objectives=[
                    'Learn about multi-class classification',
                    'Understand softmax activation',
                    'See one-hot encoding in action'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='multiclass_output',
                    show_result_after=True
                ),
                hints=[
                    'Each quadrant is a different class',
                    'The network outputs probabilities for each class',
                    'Softmax ensures outputs sum to 1'
                ]
            ),
            PathStep(
                step_number=7,
                problem_id='blobs',
                title='Design for 5-Class Classification',
                learning_objectives=[
                    'Scale to more classes',
                    'Handle overlapping class regions',
                    'Learn about classification confidence'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=1,
                    max_layers=3,
                    min_hidden_neurons=4,
                    max_hidden_neurons=16,
                    must_have_hidden=True,
                    success_message='Good design for 5 classes! Let\'s train it!'
                ),
                hints=[
                    '5 separate clusters in 2D space',
                    'Some regions may have lower confidence',
                    'The network learns probability distributions'
                ]
            ),
            PathStep(
                step_number=8,
                problem_id='patterns',
                title='What Features Distinguish Patterns?',
                learning_objectives=[
                    'Recognize sequential patterns',
                    'Understand pattern classification',
                    'Learn about feature representation'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='pattern_features',
                    show_result_after=True
                ),
                hints=[
                    'Different signal patterns to classify',
                    'The network learns pattern features',
                    'This is like basic time-series classification'
                ]
            ),
            PathStep(
                step_number=9,
                problem_id='colors',
                title='How Many Outputs for 6 Colors?',
                learning_objectives=[
                    'Classify based on continuous features',
                    'Understand color space representation',
                    'Master 6-class problems'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='color_classes',
                    show_result_after=True
                ),
                hints=[
                    '6 different color classes',
                    'RGB values as inputs',
                    'The network learns color boundaries'
                ]
            )
        ]
    ),

    'advanced-challenges': LearningPath(
        id='advanced-challenges',
        name='Advanced Challenges',
        description='Tackle depth, gradients, and complex boundaries',
        difficulty='advanced',
        estimated_time='3-4 hours',
        badge_icon='🚀',
        badge_title='Challenge Champion',
        badge_color='#EF4444',
        prerequisites=['training-mastery', 'boundaries-and-classes'],
        steps=[
            PathStep(
                step_number=1,
                problem_id='fail_xor_no_hidden',
                title='Debug: Network Can\'t Learn XOR',
                learning_objectives=[
                    'Understand capacity limitations',
                    'Learn when architecture matters',
                    'See fundamental limitations'
                ],
                required_accuracy=0.0,
                step_type='debug_challenge',
                debug_challenge=DebugChallengeData(
                    challenge_id='xor_no_hidden'
                ),
                hints=[
                    'The network has no hidden layers',
                    'What kind of patterns can a single neuron learn?'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='donut',
                title='Design for the Donut',
                learning_objectives=[
                    'Handle complex topological patterns',
                    'Master non-convex boundaries',
                    'Understand advanced decision surfaces'
                ],
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=1,
                    max_layers=4,
                    min_hidden_neurons=4,
                    max_hidden_neurons=16,
                    must_have_hidden=True,
                    success_message='Let\'s see if your architecture can handle the donut!'
                ),
                hints=[
                    'One class forms a ring around another',
                    'This requires a complex boundary shape',
                    'Test your network\'s flexibility'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='spiral',
                title='Deep vs Shallow for Spirals',
                learning_objectives=[
                    'Tackle extremely complex patterns',
                    'Learn about deep network requirements',
                    'Master challenging benchmarks'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='deep_vs_shallow',
                    show_result_after=True
                ),
                hints=[
                    'Two spirals intertwined',
                    'One of the hardest 2D problems',
                    'May need a deeper or wider network'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='fail_vanishing',
                title='Debug: Vanishing Gradients',
                learning_objectives=[
                    'Understand gradient flow issues',
                    'Learn about deep network problems',
                    'See why activation choice matters'
                ],
                required_accuracy=0.0,
                step_type='debug_challenge',
                debug_challenge=DebugChallengeData(
                    challenge_id='vanishing_gradients'
                ),
                hints=[
                    'Sigmoid activation squashes values to (0, 1)',
                    'What happens to gradients through many layers?',
                    'Consider alternative activation functions'
                ]
            ),
            PathStep(
                step_number=5,
                problem_id='fail_underfit',
                title='Fix Underfitting: Build Bigger',
                learning_objectives=[
                    'Understand model capacity',
                    'Learn about complexity matching',
                    'See when networks are too simple'
                ],
                required_accuracy=0.0,
                step_type='build_challenge',
                build_challenge=BuildChallengeData(
                    min_layers=2,
                    max_layers=5,
                    min_hidden_neurons=8,
                    max_hidden_neurons=32,
                    must_have_hidden=True,
                    success_message='Bigger network! Let\'s see if it can handle the complexity!'
                ),
                hints=[
                    'The current network is too small for this problem',
                    'Try adding more layers and neurons',
                    'Sometimes bigger IS better'
                ]
            ),
            PathStep(
                step_number=6,
                problem_id='surface',
                title='Optimize 3D Surface Training',
                learning_objectives=[
                    'Handle multi-dimensional regression',
                    'Master epoch budget optimization',
                    'Learn when to stop training'
                ],
                step_type='tuning_challenge',
                tuning_challenge=TuningChallengeData(
                    parameter='epoch_budget',
                    target_loss=0.1,
                    max_attempts=5
                ),
                hints=[
                    'The network learns a 2D surface',
                    'More epochs helps but has diminishing returns',
                    'Find the minimum epochs that reach the target loss'
                ]
            ),
            PathStep(
                step_number=7,
                problem_id='spiral',
                title='Tune Spiral Network Capacity',
                learning_objectives=[
                    'Master network capacity decisions',
                    'Find the right size for complex problems',
                    'Understand the capacity-complexity tradeoff'
                ],
                step_type='tuning_challenge',
                tuning_challenge=TuningChallengeData(
                    parameter='capacity_tuning',
                    target_loss=0.1,
                    max_attempts=5
                ),
                hints=[
                    'Spirals are among the hardest 2D problems',
                    'You need enough neurons to learn the complex boundary',
                    'Too many neurons can also cause issues'
                ]
            )
        ]
    ),

    'convolutional-vision': LearningPath(
        id='convolutional-vision',
        name='Convolutional Vision',
        description='Understand CNNs for spatial data',
        difficulty='advanced',
        estimated_time='2-3 hours',
        badge_icon='👁️',
        badge_title='Vision Virtuoso',
        badge_color='#F59E0B',
        prerequisites=['boundaries-and-classes'],
        steps=[
            PathStep(
                step_number=1,
                problem_id='shapes',
                title='Why Are CNNs Better for Images?',
                learning_objectives=[
                    'Understand convolutional layers',
                    'Learn about spatial feature extraction',
                    'See how CNNs process images'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='cnn_advantage',
                    show_result_after=True
                ),
                hints=[
                    'Draw shapes on the 8x8 grid',
                    'CNNs learn spatial patterns automatically',
                    'Watch the feature maps to see what it learns'
                ]
            ),
            PathStep(
                step_number=2,
                problem_id='digits',
                title='CNN Digit Recognition Challenge',
                learning_objectives=[
                    'Master complex spatial patterns',
                    'Handle 10-class classification',
                    'Understand deep CNN architectures'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='cnn_digit_challenge',
                    show_result_after=True
                ),
                hints=[
                    'Draw digits 0-9 on the grid',
                    'This is like mini-MNIST',
                    'CNNs excel at this task'
                ]
            ),
            PathStep(
                step_number=3,
                problem_id='patterns',
                title='CNN vs Dense Networks',
                learning_objectives=[
                    'Apply CNNs to sequential data',
                    'Understand when CNNs help vs dense networks',
                    'Master advanced pattern recognition'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='cnn_vs_dense',
                    show_result_after=True
                ),
                hints=[
                    'Different pattern types',
                    'CNNs can handle time-series too',
                    'Feature extraction + classification'
                ]
            ),
            PathStep(
                step_number=4,
                problem_id='digits',
                title='What Makes 10-Class Digits Hard?',
                learning_objectives=[
                    'Master complex CNN tasks',
                    'Handle many classes efficiently',
                    'Understand production-ready models'
                ],
                step_type='prediction_quiz',
                prediction_quiz=PredictionQuizData(
                    quiz_id='digit_architecture',
                    show_result_after=True
                ),
                hints=[
                    'This is close to real-world problems',
                    'CNNs excel at image classification',
                    'Feature extraction is key'
                ]
            ),
            PathStep(
                step_number=5,
                problem_id='digits',
                title='Find CNN Training Epochs',
                learning_objectives=[
                    'Optimize CNN training budget',
                    'Understand CNN convergence characteristics',
                    'Learn how feature reuse affects training'
                ],
                step_type='tuning_challenge',
                tuning_challenge=TuningChallengeData(
                    parameter='cnn_epoch_budget',
                    target_loss=0.3,
                    max_attempts=5
                ),
                hints=[
                    'CNNs learn features progressively',
                    'Early epochs learn edges, later epochs learn shapes',
                    'Try 30-50 epochs as a starting point'
                ]
            )
        ]
    ),
}


def get_all_paths() -> List[Dict]:
    """Get all learning paths as dictionaries"""
    return [path.to_dict() for path in LEARNING_PATHS.values()]


def get_path(path_id: str) -> Dict:
    """Get a specific learning path with step details"""
    if path_id not in LEARNING_PATHS:
        return None

    path = LEARNING_PATHS[path_id]
    result = path.to_dict()
    result['steps'] = path.get_step_details()  # Override count with detailed array
    return result
