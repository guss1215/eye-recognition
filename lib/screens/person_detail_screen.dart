import 'dart:io';
import 'package:flutter/material.dart';
import '../models/person.dart';

class PersonDetailScreen extends StatelessWidget {
  final Person person;

  const PersonDetailScreen({super.key, required this.person});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text(person.fullName),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Iris image
            if (person.irisImagePath != null)
              Container(
                height: 200,
                margin: const EdgeInsets.only(bottom: 16),
                decoration: BoxDecoration(
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: Theme.of(context).colorScheme.outline,
                  ),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.file(
                    File(person.irisImagePath!),
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => const Center(
                      child: Icon(Icons.broken_image, size: 48),
                    ),
                  ),
                ),
              ),

            // Status badge
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: person.irisTemplates != null && person.irisTemplates!.isNotEmpty
                    ? Colors.green.shade50
                    : Colors.orange.shade50,
                borderRadius: BorderRadius.circular(8),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Icon(
                    person.irisTemplates != null && person.irisTemplates!.isNotEmpty
                        ? Icons.verified
                        : Icons.warning_amber,
                    size: 18,
                    color: person.irisTemplates != null && person.irisTemplates!.isNotEmpty
                        ? Colors.green.shade700
                        : Colors.orange.shade700,
                  ),
                  const SizedBox(width: 8),
                  Text(
                    person.irisTemplates != null && person.irisTemplates!.isNotEmpty
                        ? 'Iris registered (${person.irisTemplates!.length} template${person.irisTemplates!.length > 1 ? 's' : ''})'
                        : 'No iris data',
                    style: TextStyle(
                      color: person.irisTemplates != null && person.irisTemplates!.isNotEmpty
                          ? Colors.green.shade700
                          : Colors.orange.shade700,
                      fontWeight: FontWeight.w500,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),

            _InfoTile(icon: Icons.person, label: 'Name', value: person.fullName),
            _InfoTile(
                icon: Icons.cake, label: 'Age', value: '${person.age} years'),
            if (person.email != null)
              _InfoTile(icon: Icons.email, label: 'Email', value: person.email!),
            if (person.phone != null)
              _InfoTile(icon: Icons.phone, label: 'Phone', value: person.phone!),
            if (person.notes != null)
              _InfoTile(icon: Icons.notes, label: 'Notes', value: person.notes!),

            const Divider(height: 32),
            Text(
              'Registered: ${_formatDate(person.createdAt)}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            Text(
              'Last updated: ${_formatDate(person.updatedAt)}',
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ],
        ),
      ),
    );
  }

  String _formatDate(DateTime date) {
    return '${date.day}/${date.month}/${date.year} ${date.hour}:${date.minute.toString().padLeft(2, '0')}';
  }
}

class _InfoTile extends StatelessWidget {
  final IconData icon;
  final String label;
  final String value;

  const _InfoTile({
    required this.icon,
    required this.label,
    required this.value,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 22, color: Theme.of(context).colorScheme.primary),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                    style: Theme.of(context).textTheme.labelSmall?.copyWith(
                          color: Theme.of(context).colorScheme.onSurfaceVariant,
                        )),
                Text(value, style: Theme.of(context).textTheme.bodyLarge),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
