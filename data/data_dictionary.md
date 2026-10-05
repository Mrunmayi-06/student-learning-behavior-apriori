# Member 1 Data Dictionary

## Observation period

Early-stage behaviour is defined as **days 0-27**, corresponding to the first four weeks of the course. OULAD activity dates are measured relative to the course presentation start.

## Raw / joined numerical features

| Feature | Meaning |
|---|---|
| `total_clicks` | Total VLE clicks during days 0-27. |
| `active_days` | Number of distinct days on which the student had VLE activity during days 0-27. |
| `forum_activity` | Sum of clicks on VLE activities whose `activity_type` is `forumng`. |
| `quiz_activity` | Sum of clicks on `quiz` and `externalquiz` VLE activities. |
| `resource_activity` | Sum of clicks on resource-oriented VLE activity types: resource, subpage, oucontent, url, page, folder, htmlactivity, sharedsubpage, and repeatactivity. |
| `assignment_timing` | Timing category for early TMA/CMA submissions relative to the assessment due date. |

## Categorical transaction items

Numerical behaviour is converted into Low/Medium/High using student rank tertiles.

| Item | Meaning |
|---|---|
| `Forum_Low/Medium/High` | Relative level of early forum activity. |
| `Quiz_Low/Medium/High` | Relative level of early quiz activity. |
| `Resource_Low/Medium/High` | Relative level of early resource activity. |
| `ActiveDays_Low/Medium/High` | Relative number of early active days. |
| `Assignment_Early` | Average early TMA/CMA submission date was before the due date. |
| `Assignment_OnTime` | Average submission was on the due date or within one day after it. |
| `Assignment_Late` | Average submission was more than one day after the due date. |
| `Assignment_Missing` | At least one scheduled early TMA/CMA had no recorded submission. |
| `Engagement_Low/Medium/High` | Overall early engagement category based on rank-derived levels of clicks, active days, forum, quiz, and resource activity. |
| `Result_Pass` | OULAD final result was Pass or Distinction. Distinction is grouped with Pass to match the project's three requested outcome categories. |
| `Result_Fail` | OULAD final result was Fail. |
| `Result_Withdrawn` | OULAD final result was Withdrawn. |

## Tables used

- `studentInfo.csv`: student/course/presentation identity and final result.
- `studentVle.csv`: student VLE interactions and dates.
- `vle.csv`: VLE resource/activity type lookup.
- `studentAssessment.csv`: student assessment submissions.
- `assessments.csv`: assessment type and due date.
- `studentRegistration.csv`: registration/unregistration context.
- `courses.csv`: module presentation duration.

## Transaction design

There is one transaction per student-course-presentation record. Each transaction contains early behavioural items and exactly one final outcome item.

Example:

`Forum_Low,Quiz_Low,Resource_High,ActiveDays_Low,Assignment_Late,Engagement_Low,Result_Withdrawn`
