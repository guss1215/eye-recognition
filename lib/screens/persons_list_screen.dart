import 'package:flutter/material.dart';
import '../models/person.dart';
import '../services/database_service.dart';
import 'person_detail_screen.dart';
import 'registration_screen.dart';

class PersonsListScreen extends StatefulWidget {
  const PersonsListScreen({super.key});

  @override
  State<PersonsListScreen> createState() => _PersonsListScreenState();
}

class _PersonsListScreenState extends State<PersonsListScreen> {
  final DatabaseService _dbService = DatabaseService();
  final _searchController = TextEditingController();
  List<Person> _persons = [];
  bool _isLoading = true;

  @override
  void initState() {
    super.initState();
    _loadPersons();
  }

  Future<void> _loadPersons() async {
    setState(() => _isLoading = true);
    try {
      final persons = _searchController.text.isEmpty
          ? await _dbService.getAllPersons()
          : await _dbService.searchPersons(_searchController.text);
      setState(() {
        _persons = persons;
        _isLoading = false;
      });
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  @override
  void dispose() {
    _searchController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Registered Persons'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Column(
        children: [
          Padding(
            padding: const EdgeInsets.all(12),
            child: TextField(
              controller: _searchController,
              decoration: InputDecoration(
                hintText: 'Search by name or email...',
                prefixIcon: const Icon(Icons.search),
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
                suffixIcon: _searchController.text.isNotEmpty
                    ? IconButton(
                        icon: const Icon(Icons.clear),
                        onPressed: () {
                          _searchController.clear();
                          _loadPersons();
                        },
                      )
                    : null,
              ),
              onChanged: (_) => _loadPersons(),
            ),
          ),
          Expanded(
            child: _isLoading
                ? const Center(child: CircularProgressIndicator())
                : _persons.isEmpty
                    ? Center(
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(Icons.people_outline,
                                size: 64,
                                color: Theme.of(context)
                                    .colorScheme
                                    .onSurfaceVariant),
                            const SizedBox(height: 16),
                            Text(
                              'No persons registered yet',
                              style: Theme.of(context).textTheme.titleMedium,
                            ),
                          ],
                        ),
                      )
                    : RefreshIndicator(
                        onRefresh: _loadPersons,
                        child: ListView.builder(
                          itemCount: _persons.length,
                          itemBuilder: (context, index) {
                            final person = _persons[index];
                            return ListTile(
                              leading: CircleAvatar(
                                backgroundColor:
                                    Theme.of(context).colorScheme.primaryContainer,
                                child: Text(
                                  person.firstName[0].toUpperCase(),
                                  style: TextStyle(
                                    color: Theme.of(context)
                                        .colorScheme
                                        .onPrimaryContainer,
                                  ),
                                ),
                              ),
                              title: Text(person.fullName),
                              subtitle: Text(
                                '${person.age} years${person.email != null ? ' • ${person.email}' : ''}',
                              ),
                              trailing: Icon(
                                person.irisTemplate != null
                                    ? Icons.visibility
                                    : Icons.visibility_off,
                                color: person.irisTemplate != null
                                    ? Colors.green
                                    : Colors.grey,
                                size: 20,
                              ),
                              onTap: () {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) =>
                                        PersonDetailScreen(person: person),
                                  ),
                                ).then((_) => _loadPersons());
                              },
                            );
                          },
                        ),
                      ),
          ),
        ],
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (context) => const RegistrationScreen(),
            ),
          ).then((_) => _loadPersons());
        },
        child: const Icon(Icons.person_add),
      ),
    );
  }
}
