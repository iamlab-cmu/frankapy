# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: feedback_controller_params_msg.proto

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='feedback_controller_params_msg.proto',
  package='',
  syntax='proto2',
  serialized_options=None,
  serialized_pb=b'\n$feedback_controller_params_msg.proto\"p\n+CartesianImpedanceFeedbackControllerMessage\x12!\n\x19translational_stiffnesses\x18\x01 \x03(\x01\x12\x1e\n\x16rotational_stiffnesses\x18\x02 \x03(\x01\"q\n\"ForceAxisFeedbackControllerMessage\x12\x1f\n\x17translational_stiffness\x18\x01 \x02(\x01\x12\x1c\n\x14rotational_stiffness\x18\x02 \x02(\x01\x12\x0c\n\x04\x61xis\x18\x03 \x03(\x01\"K\n\'JointImpedanceFeedbackControllerMessage\x12\x0f\n\x07k_gains\x18\x01 \x03(\x01\x12\x0f\n\x07\x64_gains\x18\x02 \x03(\x01\"d\n*InternalImpedanceFeedbackControllerMessage\x12\x1c\n\x14\x63\x61rtesian_impedances\x18\x01 \x03(\x01\x12\x18\n\x10joint_impedances\x18\x02 \x03(\x01'
)




_CARTESIANIMPEDANCEFEEDBACKCONTROLLERMESSAGE = _descriptor.Descriptor(
  name='CartesianImpedanceFeedbackControllerMessage',
  full_name='CartesianImpedanceFeedbackControllerMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='translational_stiffnesses', full_name='CartesianImpedanceFeedbackControllerMessage.translational_stiffnesses', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rotational_stiffnesses', full_name='CartesianImpedanceFeedbackControllerMessage.rotational_stiffnesses', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=40,
  serialized_end=152,
)


_FORCEAXISFEEDBACKCONTROLLERMESSAGE = _descriptor.Descriptor(
  name='ForceAxisFeedbackControllerMessage',
  full_name='ForceAxisFeedbackControllerMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='translational_stiffness', full_name='ForceAxisFeedbackControllerMessage.translational_stiffness', index=0,
      number=1, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='rotational_stiffness', full_name='ForceAxisFeedbackControllerMessage.rotational_stiffness', index=1,
      number=2, type=1, cpp_type=5, label=2,
      has_default_value=False, default_value=float(0),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='axis', full_name='ForceAxisFeedbackControllerMessage.axis', index=2,
      number=3, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=154,
  serialized_end=267,
)


_JOINTIMPEDANCEFEEDBACKCONTROLLERMESSAGE = _descriptor.Descriptor(
  name='JointImpedanceFeedbackControllerMessage',
  full_name='JointImpedanceFeedbackControllerMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='k_gains', full_name='JointImpedanceFeedbackControllerMessage.k_gains', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='d_gains', full_name='JointImpedanceFeedbackControllerMessage.d_gains', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=269,
  serialized_end=344,
)


_INTERNALIMPEDANCEFEEDBACKCONTROLLERMESSAGE = _descriptor.Descriptor(
  name='InternalImpedanceFeedbackControllerMessage',
  full_name='InternalImpedanceFeedbackControllerMessage',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='cartesian_impedances', full_name='InternalImpedanceFeedbackControllerMessage.cartesian_impedances', index=0,
      number=1, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
    _descriptor.FieldDescriptor(
      name='joint_impedances', full_name='InternalImpedanceFeedbackControllerMessage.joint_impedances', index=1,
      number=2, type=1, cpp_type=5, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, file=DESCRIPTOR),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto2',
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=346,
  serialized_end=446,
)

DESCRIPTOR.message_types_by_name['CartesianImpedanceFeedbackControllerMessage'] = _CARTESIANIMPEDANCEFEEDBACKCONTROLLERMESSAGE
DESCRIPTOR.message_types_by_name['ForceAxisFeedbackControllerMessage'] = _FORCEAXISFEEDBACKCONTROLLERMESSAGE
DESCRIPTOR.message_types_by_name['JointImpedanceFeedbackControllerMessage'] = _JOINTIMPEDANCEFEEDBACKCONTROLLERMESSAGE
DESCRIPTOR.message_types_by_name['InternalImpedanceFeedbackControllerMessage'] = _INTERNALIMPEDANCEFEEDBACKCONTROLLERMESSAGE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

CartesianImpedanceFeedbackControllerMessage = _reflection.GeneratedProtocolMessageType('CartesianImpedanceFeedbackControllerMessage', (_message.Message,), {
  'DESCRIPTOR' : _CARTESIANIMPEDANCEFEEDBACKCONTROLLERMESSAGE,
  '__module__' : 'feedback_controller_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:CartesianImpedanceFeedbackControllerMessage)
  })
_sym_db.RegisterMessage(CartesianImpedanceFeedbackControllerMessage)

ForceAxisFeedbackControllerMessage = _reflection.GeneratedProtocolMessageType('ForceAxisFeedbackControllerMessage', (_message.Message,), {
  'DESCRIPTOR' : _FORCEAXISFEEDBACKCONTROLLERMESSAGE,
  '__module__' : 'feedback_controller_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:ForceAxisFeedbackControllerMessage)
  })
_sym_db.RegisterMessage(ForceAxisFeedbackControllerMessage)

JointImpedanceFeedbackControllerMessage = _reflection.GeneratedProtocolMessageType('JointImpedanceFeedbackControllerMessage', (_message.Message,), {
  'DESCRIPTOR' : _JOINTIMPEDANCEFEEDBACKCONTROLLERMESSAGE,
  '__module__' : 'feedback_controller_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:JointImpedanceFeedbackControllerMessage)
  })
_sym_db.RegisterMessage(JointImpedanceFeedbackControllerMessage)

InternalImpedanceFeedbackControllerMessage = _reflection.GeneratedProtocolMessageType('InternalImpedanceFeedbackControllerMessage', (_message.Message,), {
  'DESCRIPTOR' : _INTERNALIMPEDANCEFEEDBACKCONTROLLERMESSAGE,
  '__module__' : 'feedback_controller_params_msg_pb2'
  # @@protoc_insertion_point(class_scope:InternalImpedanceFeedbackControllerMessage)
  })
_sym_db.RegisterMessage(InternalImpedanceFeedbackControllerMessage)


# @@protoc_insertion_point(module_scope)