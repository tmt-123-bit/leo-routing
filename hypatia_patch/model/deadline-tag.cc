#include "deadline-tag.h"

namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (DeadlineTag);

TypeId
DeadlineTag::GetTypeId (void)
{
    static TypeId tid = TypeId ("ns3::DeadlineTag")
        .SetParent<Tag> ()
        .SetGroupName ("SatelliteNetwork")
        .AddConstructor<DeadlineTag> ()
    ;
    return tid;
}

TypeId
DeadlineTag::GetInstanceTypeId (void) const
{
    return GetTypeId ();
}

uint32_t
DeadlineTag::GetSerializedSize (void) const
{
    return 4 + 8 + 4 + 4 + 8 + 1;
}

void
DeadlineTag::Serialize (TagBuffer buffer) const
{
    buffer.WriteU32 (m_dst_node_id);
    buffer.WriteU64 (m_deadline_ns);
    buffer.WriteU32 (m_burst_id);
    buffer.WriteU32 (m_seq);
    buffer.WriteU64 (m_send_ns);
    buffer.WriteU8 (m_traffic_class);
}

void
DeadlineTag::Deserialize (TagBuffer buffer)
{
    m_dst_node_id = buffer.ReadU32 ();
    m_deadline_ns = buffer.ReadU64 ();
    m_burst_id = buffer.ReadU32 ();
    m_seq = buffer.ReadU32 ();
    m_send_ns = buffer.ReadU64 ();
    m_traffic_class = buffer.ReadU8 ();
}

void
DeadlineTag::Print (std::ostream &os) const
{
    os << "DeadlineTag(dst=" << m_dst_node_id
       << ", deadline_ns=" << m_deadline_ns
       << ", burst=" << m_burst_id << ", seq=" << m_seq << ")";
}

DeadlineTag::DeadlineTag ()
    : m_dst_node_id (0), m_deadline_ns (0), m_burst_id (0), m_seq (0),
      m_send_ns (0), m_traffic_class (0)
{}

DeadlineTag::DeadlineTag (uint32_t dst_node_id, uint64_t deadline_ns,
                          uint32_t burst_id, uint32_t seq, uint64_t send_ns,
                          uint8_t traffic_class)
    : m_dst_node_id (dst_node_id), m_deadline_ns (deadline_ns),
      m_burst_id (burst_id), m_seq (seq), m_send_ns (send_ns),
      m_traffic_class (traffic_class)
{}

uint32_t DeadlineTag::GetDstNodeId () const { return m_dst_node_id; }
uint64_t DeadlineTag::GetDeadlineNs () const { return m_deadline_ns; }
uint32_t DeadlineTag::GetBurstId () const { return m_burst_id; }
uint32_t DeadlineTag::GetSeq () const { return m_seq; }
uint64_t DeadlineTag::GetSendNs () const { return m_send_ns; }
uint8_t DeadlineTag::GetTrafficClass () const { return m_traffic_class; }

} // namespace ns3
