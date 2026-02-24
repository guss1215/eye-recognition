import 'dart:convert';

class Person {
  final String id;
  final String firstName;
  final String lastName;
  final int age;
  final String? email;
  final String? phone;
  final String? notes;
  final String? irisImagePath;
  final List<List<double>>? irisTemplates; // multiple encoded iris templates
  final DateTime createdAt;
  final DateTime updatedAt;

  Person({
    required this.id,
    required this.firstName,
    required this.lastName,
    required this.age,
    this.email,
    this.phone,
    this.notes,
    this.irisImagePath,
    this.irisTemplates,
    required this.createdAt,
    required this.updatedAt,
  });

  /// Backward-compatible getter: returns the first (best) template.
  List<double>? get irisTemplate =>
      irisTemplates?.isNotEmpty == true ? irisTemplates!.first : null;

  Map<String, dynamic> toMap() {
    return {
      'id': id,
      'first_name': firstName,
      'last_name': lastName,
      'age': age,
      'email': email,
      'phone': phone,
      'notes': notes,
      'iris_image_path': irisImagePath,
      'iris_templates': irisTemplates != null ? jsonEncode(irisTemplates) : null,
      'created_at': createdAt.toIso8601String(),
      'updated_at': updatedAt.toIso8601String(),
    };
  }

  factory Person.fromMap(Map<String, dynamic> map) {
    return Person(
      id: map['id'] as String,
      firstName: map['first_name'] as String,
      lastName: map['last_name'] as String,
      age: map['age'] as int,
      email: map['email'] as String?,
      phone: map['phone'] as String?,
      notes: map['notes'] as String?,
      irisImagePath: map['iris_image_path'] as String?,
      irisTemplates: _parseTemplates(map),
      createdAt: DateTime.parse(map['created_at'] as String),
      updatedAt: DateTime.parse(map['updated_at'] as String),
    );
  }

  /// Parses templates from either v2 (JSON array) or v1 (comma-separated) format.
  static List<List<double>>? _parseTemplates(Map<String, dynamic> map) {
    // v2 format: JSON array of arrays in iris_templates column
    if (map['iris_templates'] != null) {
      final decoded = jsonDecode(map['iris_templates'] as String);
      return (decoded as List)
          .map((t) =>
              (t as List).map((e) => (e as num).toDouble()).toList())
          .toList();
    }
    // v1 format: single comma-separated template in iris_template column
    if (map['iris_template'] != null) {
      final single = (map['iris_template'] as String)
          .split(',')
          .map((e) => double.parse(e))
          .toList();
      return [single];
    }
    return null;
  }

  Person copyWith({
    String? firstName,
    String? lastName,
    int? age,
    String? email,
    String? phone,
    String? notes,
    String? irisImagePath,
    List<List<double>>? irisTemplates,
  }) {
    return Person(
      id: id,
      firstName: firstName ?? this.firstName,
      lastName: lastName ?? this.lastName,
      age: age ?? this.age,
      email: email ?? this.email,
      phone: phone ?? this.phone,
      notes: notes ?? this.notes,
      irisImagePath: irisImagePath ?? this.irisImagePath,
      irisTemplates: irisTemplates ?? this.irisTemplates,
      createdAt: createdAt,
      updatedAt: DateTime.now(),
    );
  }

  String get fullName => '$firstName $lastName';
}
