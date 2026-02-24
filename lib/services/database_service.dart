import 'dart:convert';
import 'package:sqflite/sqflite.dart';
import 'package:path/path.dart';
import '../models/person.dart';

class DatabaseService {
  static Database? _database;

  Future<Database> get database async {
    if (_database != null) return _database!;
    _database = await _initDatabase();
    return _database!;
  }

  Future<Database> _initDatabase() async {
    final dbPath = await getDatabasesPath();
    final path = join(dbPath, 'eye_recognition.db');

    return await openDatabase(
      path,
      version: 2,
      onCreate: _onCreate,
      onUpgrade: _onUpgrade,
    );
  }

  Future<void> _onCreate(Database db, int version) async {
    await db.execute('''
      CREATE TABLE persons (
        id TEXT PRIMARY KEY,
        first_name TEXT NOT NULL,
        last_name TEXT NOT NULL,
        age INTEGER NOT NULL,
        email TEXT,
        phone TEXT,
        notes TEXT,
        iris_image_path TEXT,
        iris_templates TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
      )
    ''');
  }

  Future<void> _onUpgrade(Database db, int oldVersion, int newVersion) async {
    if (oldVersion < 2) {
      // Add the new multi-template column
      await db.execute('ALTER TABLE persons ADD COLUMN iris_templates TEXT');

      // Migrate existing single templates to JSON array format
      final rows = await db.query(
        'persons',
        columns: ['id', 'iris_template'],
        where: 'iris_template IS NOT NULL',
      );
      for (final row in rows) {
        final oldTemplate = row['iris_template'] as String;
        final templateList =
            oldTemplate.split(',').map((e) => double.parse(e)).toList();
        final jsonTemplates = jsonEncode([templateList]);
        await db.update(
          'persons',
          {'iris_templates': jsonTemplates},
          where: 'id = ?',
          whereArgs: [row['id']],
        );
      }
    }
  }

  Future<String> insertPerson(Person person) async {
    final db = await database;
    await db.insert('persons', person.toMap());
    return person.id;
  }

  Future<Person?> getPersonById(String id) async {
    final db = await database;
    final maps = await db.query('persons', where: 'id = ?', whereArgs: [id]);
    if (maps.isEmpty) return null;
    return Person.fromMap(maps.first);
  }

  Future<List<Person>> getAllPersons() async {
    final db = await database;
    final maps = await db.query('persons', orderBy: 'created_at DESC');
    return maps.map((map) => Person.fromMap(map)).toList();
  }

  Future<void> updatePerson(Person person) async {
    final db = await database;
    await db.update(
      'persons',
      person.toMap(),
      where: 'id = ?',
      whereArgs: [person.id],
    );
  }

  Future<void> deletePerson(String id) async {
    final db = await database;
    await db.delete('persons', where: 'id = ?', whereArgs: [id]);
  }

  Future<List<Person>> searchPersons(String query) async {
    final db = await database;
    final maps = await db.query(
      'persons',
      where: 'first_name LIKE ? OR last_name LIKE ? OR email LIKE ?',
      whereArgs: ['%$query%', '%$query%', '%$query%'],
    );
    return maps.map((map) => Person.fromMap(map)).toList();
  }

  /// Returns all persons that have iris templates stored.
  Future<List<Person>> getPersonsWithIrisTemplate() async {
    final db = await database;
    final maps = await db.query(
      'persons',
      where: 'iris_templates IS NOT NULL',
    );
    return maps.map((map) => Person.fromMap(map)).toList();
  }
}
