/*
 * Copyright (c) 2020 ETH Zurich
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation;
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
 *
 * Author: Simon               2020
 */

#ifndef ARBITER_SINGLE_FORWARD_HELPER
#define ARBITER_SINGLE_FORWARD_HELPER

#include "ns3/ipv4-routing-helper.h"
#include "ns3/basic-simulation.h"
#include "ns3/topology-satellite-network.h"
#include "ns3/ipv4-arbiter-routing.h"
#include "ns3/arbiter-single-forward.h"
#include "ns3/abort.h"
#include <map>

namespace ns3 {

    class ArbiterSingleForwardHelper
    {
    public:
        ArbiterSingleForwardHelper(Ptr<BasicSimulation> basicSimulation, NodeContainer nodes);
    private:
        std::vector<std::vector<std::tuple<int32_t, int32_t, int32_t>>> InitialEmptyForwardingState();
        void UpdateForwardingState(int64_t t);

        // Bridge routing extension (satellite_network_routing_mode=bridge):
        // forwarding state comes from an external policy server instead of
        // the fstate files (files remain the fallback base on the server side).
        void BridgeUpdateForwardingState(int64_t t);
        void BridgeSend(const std::string& line);
        std::string BridgeRecvLine();
        std::pair<int32_t, int32_t> ResolveIslIf(int32_t u, int32_t v);

        // Parameters
        Ptr<BasicSimulation> m_basicSimulation;
        NodeContainer m_nodes;
        int64_t m_dynamicStateUpdateIntervalNs;
        std::vector<Ptr<ArbiterSingleForward>> m_arbiters;
        bool m_bridgeMode = false;
        int m_bridgeSock = -1;

    };

} // namespace ns3

#endif /* ARBITER_SINGLE_FORWARD_HELPER */
