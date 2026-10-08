#include "deadline-udp-scheduler.h"
#include "../model/deadline-udp-application.h"

#include <fstream>
#include <sstream>
#include <map>

namespace ns3 {

DeadlineUdpScheduler::DeadlineUdpScheduler (
        Ptr<BasicSimulation> basicSimulation, NodeContainer nodes)
{
    std::string run_dir = basicSimulation->GetRunDir ();
    std::string schedule_file = run_dir + "/"
        + basicSimulation->GetConfigParamOrFail ("deadline_udp_burst_schedule_filename");

    std::ifstream fs (schedule_file);
    NS_ABORT_MSG_IF (!fs.is_open (), "Deadline UDP schedule file could not be opened");

    std::map<int32_t, Ptr<DeadlineUdpApplication> > appOfNode;

    std::string line;
    int line_nr = 0;
    while (std::getline (fs, line)) {
        line_nr++;
        if (line.empty () || line_nr == 1) {
            continue;  // header
        }
        std::stringstream ss (line);
        std::string cell;
        std::vector<std::string> f;
        while (std::getline (ss, cell, ',')) {
            f.push_back (cell);
        }
        NS_ABORT_MSG_IF (f.size () < 7, "Invalid deadline UDP schedule row at line "
                         << line_nr);
        uint8_t traffic_class = f.size () >= 8
            ? (uint8_t) std::stoul (f[7]) : (uint8_t) 0;

        uint32_t burst_id = (uint32_t) std::stoll (f[0]);
        int64_t start_time_ns = std::stoll (f[1]);
        int32_t from_node_id = (int32_t) std::stoll (f[2]);
        int32_t to_node_id = (int32_t) std::stoll (f[3]);
        double target_rate_mbps = std::stod (f[4]);
        int64_t duration_ns = std::stoll (f[5]);
        int64_t deadline_ms = std::stoll (f[6]);
        int64_t deadline_offset_ns = deadline_ms * 1000000L;  // per-packet deadline offset

        NS_ABORT_MSG_IF (from_node_id < 0 || (uint32_t) from_node_id >= nodes.GetN (),
                         "Invalid from_node_id in deadline UDP schedule");
        NS_ABORT_MSG_IF (to_node_id < 0 || (uint32_t) to_node_id >= nodes.GetN (),
                         "Invalid to_node_id in deadline UDP schedule");

        InetSocketAddress target = InetSocketAddress (
            nodes.Get (to_node_id)->GetObject<Ipv4> ()->GetAddress (1, 0).GetLocal (), 1726);

        for (int32_t node_id : {from_node_id, to_node_id}) {
            if (appOfNode.find (node_id) == appOfNode.end ()) {
                Ptr<DeadlineUdpApplication> app = CreateObject<DeadlineUdpApplication> ();
                app->SetLoggingPath (run_dir + "/logs_ns3");
                nodes.Get (node_id)->AddApplication (app);
                app->SetStartTime (NanoSeconds (0));
                app->SetStopTime (NanoSeconds (basicSimulation->GetSimulationEndTimeNs () - 1000000));
                appOfNode[node_id] = app;
            }
        }

        appOfNode[from_node_id]->RegisterOutgoingBurst (
            burst_id, from_node_id, to_node_id, target,
            start_time_ns, duration_ns, target_rate_mbps, deadline_offset_ns,
            traffic_class);

        std::cout << "    >> Deadline UDP burst " << burst_id << ": node "
                  << from_node_id << " -> " << to_node_id << " @ "
                  << target_rate_mbps << " Mbit/s for "
                  << (duration_ns / 1000000) << " ms, deadline "
                  << deadline_ms << " ms" << std::endl;
    }

    basicSimulation->RegisterTimestamp ("deadline_udp_bursts_installed");
}

} // namespace ns3
