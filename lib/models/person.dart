class Person {
  final String id;
  final String firstName;
  final String lastName;
  final int age;
  final String? email;
  final String? phone;
  final String? notes;
  final String? irisImagePath;
  final List<double>? irisTemplate; // encoded iris features for matching
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
    this.irisTemplate,
    required this.createdAt,
    required this.updatedAt,
  });

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
      'iris_template': irisTemplate?.join(','),
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
      irisTemplate: map['iris_template'] != null
          ? (map['iris_template'] as String)
              .split(',')
              .map((e) => double.parse(e))
              .toList()
          : null,
      createdAt: DateTime.parse(map['created_at'] as String),
      updatedAt: DateTime.parse(map['updated_at'] as String),
    );
  }

  Person copyWith({
    String? firstName,
    String? lastName,
    int? age,
    String? email,
    String? phone,
    String? notes,
    String? irisImagePath,
    List<double>? irisTemplate,
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
      irisTemplate: irisTemplate ?? this.irisTemplate,
      createdAt: createdAt,
      updatedAt: DateTime.now(),
    );
  }

  String get fullName => '$firstName $lastName';
}
