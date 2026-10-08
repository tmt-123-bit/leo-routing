#include "purge-srpf-queue.h"

namespace ns3 {

/**
 * The PurgeSrpfQueue is used as a device queue on ISL interfaces for
 * Packet items. The explicit template class instantiation below mirrors
 * drop-tail-queue.cc so that the Object factory can create it through
 * its registered TypeId.
 */

NS_OBJECT_TEMPLATE_CLASS_DEFINE (PurgeSrpfQueue, Packet);

} // namespace ns3
