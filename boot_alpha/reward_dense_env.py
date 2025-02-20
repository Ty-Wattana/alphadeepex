from alphagen.data.tokens import *
from alphagen.config import MAX_EXPR_LENGTH
import math
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnvWrapper
from alphagen.rl.env.core import AlphaEnvCore

from typing import Optional, Tuple, List

class AlphaDenseEnv(AlphaEnvCore):

    """"
    Override the step method to make a reward-dense environment.
    """
    def step(self, action: Token) -> Tuple[List[Token], float, bool, bool, dict]:
        if (isinstance(action, SequenceIndicatorToken) and
                action.indicator == SequenceIndicatorType.SEP):
            reward = self._evaluate()
            done = True
        elif len(self._tokens) < MAX_EXPR_LENGTH:
            self._tokens.append(action)
            self._builder.add_token(action)
            done = False
            reward = self._evaluate() if self._builder.is_valid() else -1.
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.

        if math.isnan(reward):
            reward = 0.

        return self._tokens, reward, done, False, self._valid_action_types()
    
def AlphaEnvDense(pool: AlphaPoolBase, subexprs: Optional[List[Expression]] = None, **kwargs):
    return AlphaEnvWrapper(AlphaDenseEnv(pool=pool, **kwargs), subexprs=subexprs)