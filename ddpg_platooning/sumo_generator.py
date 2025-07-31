import os

class SUMONetworkGenerator:
    """
    Generate SUMO configuration files for the platooning environment.
    """

    def __init__(self, output_dir: str = "sumo_files"):
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def create_network(self) -> str:
        """Create SUMO network file (.net.xml)."""
        net_file = os.path.join(self.output_dir, "highway.net.xml")

        net_content = '''<?xml version="1.0" encoding="UTF-8"?>
<net version="1.9" junctionCornerDetail="5" limitTurnSpeed="5.50" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/net_file.xsd">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,5000.00,0.00" origBoundary="-10000000000.00,-10000000000.00,10000000000.00,10000000000.00" projParameter="!"/>
    
    <edge id="highway" from="start" to="end" priority="1">
        <lane id="highway_0" index="0" speed="50.00" length="5000.00" shape="0.00,-1.60 5000.00,-1.60"/>
    </edge>
    
    <junction id="start" type="dead_end" x="0.00" y="0.00" incLanes="" intLanes="" shape="0.00,0.00 0.00,-3.20"/>
    <junction id="end" type="dead_end" x="5000.00" y="0.00" incLanes="highway_0" intLanes="" shape="5000.00,-3.20 5000.00,0.00"/>
</net>'''

        with open(net_file, 'w') as f:
            f.write(net_content)

        return net_file

    def create_route_file(self) -> str:
        """Create SUMO route file (.rou.xml)."""
        rou_file = os.path.join(self.output_dir, "highway.rou.xml")

        rou_content = '''<?xml version="1.0" encoding="UTF-8"?>
<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">
    <vType id="leader_vehicle" accel="3.5" decel="3.5" sigma="0" length="3.2" maxSpeed="50"/>
    <vType id="platoon_vehicle" accel="3.5" decel="3.5" sigma="0" length="3.2" maxSpeed="50"/>
    
    <route id="highway_route" edges="highway"/>
</routes>'''

        with open(rou_file, 'w') as f:
            f.write(rou_content)

        return rou_file

    def create_config_file(self, net_file: str, rou_file: str) -> str:
        """Create SUMO configuration file (.sumocfg)."""
        cfg_file = os.path.join(self.output_dir, "highway.sumocfg")

        cfg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{os.path.basename(net_file)}"/>
        <route-files value="{os.path.basename(rou_file)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
        <step-length value="0.25"/>
    </time>
    <processing>
        <collision.check-junctions value="true"/>
        <collision.action value="warn"/>
    </processing>
    <report>
        <verbose value="false"/>
        <no-warnings value="true"/>
    </report>
</configuration>'''

        with open(cfg_file, 'w') as f:
            f.write(cfg_content)

        return cfg_file
