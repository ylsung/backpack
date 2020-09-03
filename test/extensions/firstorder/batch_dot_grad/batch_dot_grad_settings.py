"""Test configurations for ```backpack.core.extensions.firstorder.batch_grad_dot``.

The tests are taken from `test.extensions.firstorder.firstorder_settings`,
but additional custom tests can be defined here by appending it to the list.
"""

from test.extensions.firstorder.firstorder_settings import FIRSTORDER_SETTINGS

SHARED_SETTINGS = FIRSTORDER_SETTINGS

LOCAL_SETTING = []

BATCHDOTGRAD_SETTINGS = SHARED_SETTINGS + LOCAL_SETTING
