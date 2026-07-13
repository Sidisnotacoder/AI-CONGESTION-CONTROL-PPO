import os
import sys
from functools import partial
from mininet.net import Mininet
from mininet.link import TCLink
from mininet.node import OVSSwitch
from mininet.cli import CLI

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.network import BottleneckTopo, FAIL_MODE

topo = BottleneckTopo()

net = Mininet(
    topo=topo,
    link=TCLink,
    # IMPORTANT: controller=None, but that alone leaves OVSSwitch in
    # failMode='secure' (empty flow table, drops everything). Force
    # 'standalone' so switches behave as plain L2 learning switches.
    switch=partial(OVSSwitch, failMode=FAIL_MODE),
    controller=None
)

net.start()

CLI(net)

net.stop()


