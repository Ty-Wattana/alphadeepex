from alphagen.data.tokens import *
from alphagen.config import MAX_EXPR_LENGTH
import math
from alphagen.models.alpha_pool import AlphaPoolBase
from alphagen.rl.env.wrapper import AlphaEnvWrapper
from alphagen.rl.env.core import AlphaEnvCore

from typing import Optional, Tuple, List

import numpy as np
import re
from alphagen.config import *

SIZE_OP = len(OPERATORS)
SIZE_FEATURE = len(FeatureType)
SIZE_DELTA_TIME = len(DELTA_TIMES)
SIZE_CONSTANT = len(CONSTANTS)
SIZE_SEP = 1
SIZE_ACTION = SIZE_OP + SIZE_FEATURE + SIZE_DELTA_TIME + SIZE_CONSTANT + SIZE_SEP

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
            reward = self._evaluate() if self._builder.is_valid() else 0
        else:
            done = True
            reward = self._evaluate() if self._builder.is_valid() else -1.

        if math.isnan(reward):
            reward = 0.

        return self._tokens, reward, done, False, self._valid_action_types()
    
class AlphaStrictEnvDense(AlphaEnvWrapper):

    """"
    Override action masks to be more strict.
    """

    def action_masks(self) -> np.ndarray:
        res = np.zeros(self.size_action, dtype=bool)
        valid = self.env.valid_action_types()

        offset = 0              # Operators
        for i in range(offset, offset + SIZE_OP):
            ops = OPERATORS[i - offset]
            if valid['op'][ops.category_type()]:
                res[i] = True
                if len(self.env._tokens) >= 2:
                    last_two_op = str(self.env._tokens[-2])
                    last_op = str(self.env._tokens[-1])
                    str_ops = str_ops = re.findall(r"\.([^']+)'>", str(ops))
                    str_ops = str_ops[0].split('.')[-1]
                    if str_ops == last_two_op or ops == last_op:
                       res[i] = False

        offset += SIZE_OP
        if valid['select'][1]:  # Features
            res[offset:offset + SIZE_FEATURE] = True
        offset += SIZE_FEATURE
        if valid['select'][2]:  # Constants
            res[offset:offset + SIZE_CONSTANT] = True
        offset += SIZE_CONSTANT
        if valid['select'][3]:  # Delta time
            res[offset:offset + SIZE_DELTA_TIME] = True
        offset += SIZE_DELTA_TIME
        if valid['select'][1]:  # Sub-expressions
            res[offset:offset + len(self.subexprs)] = True
        offset += len(self.subexprs)
        if valid['select'][4]:  # SEP
            res[offset] = True
        return res
    
def AlphaEnvDense(pool: AlphaPoolBase, subexprs: Optional[List[Expression]] = None, constrain: bool = False, **kwargs):
    if constrain:
        return AlphaStrictEnvDense(AlphaDenseEnv(pool=pool, **kwargs), subexprs=subexprs)
    else:
        return AlphaEnvWrapper(AlphaDenseEnv(pool=pool, **kwargs), subexprs=subexprs)